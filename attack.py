"""
Sybil 攻擊工具模組
================

此模組包含 Sybil 攻擊的核心功能：
- 攻擊策略實現
- 聯邦學習聚合操作
- 攻擊效果評估
- 攻擊歷史追蹤

Author: Security Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from environment import FederatedLearningEnvironment

class SybilAttackOrchestrator:
    """Sybil 攻擊編排器"""
    
    def __init__(self, environment: FederatedLearningEnvironment):
        self.environment = environment
        self.attack_history = []
        self.current_round = 0
        self.attack_active = False
        self.attack_start_round = 0
        
    def federated_averaging(self, models: List[nn.Module]) -> nn.Module:
        """聯邦平均算法 - 聚合模型參數"""
        global_model = copy.deepcopy(models[0])
        
        # 計算每個模型的權重（基於數據量）
        total_data_size = sum([
            client.get_data_size() for client in 
            self.environment.honest_clients + self.environment.sybil_clients
        ])
        
        # 重置全局模型參數
        for param in global_model.parameters():
            param.data.zero_()
            
        # 聚合模型參數
        model_idx = 0
        for client in self.environment.honest_clients:
            weight = client.get_data_size() / total_data_size
            for global_param, local_param in zip(global_model.parameters(), 
                                               models[model_idx].parameters()):
                global_param.data += weight * local_param.data
            model_idx += 1
            
        # 如果攻擊激活，包含 Sybil 客戶端的更新
        if self.attack_active:
            for client in self.environment.sybil_clients:
                weight = client.get_data_size() / total_data_size
                for global_param, local_param in zip(global_model.parameters(), 
                                                   models[model_idx].parameters()):
                    global_param.data += weight * local_param.data
                model_idx += 1
                
        return global_model
    
    def start_attack(self, start_round: int = 3):
        """開始 Sybil 攻擊"""
        self.attack_start_round = start_round
        if self.current_round >= start_round:
            self.attack_active = True
            print(f"🚨 Sybil 攻擊已在第 {self.current_round} 輪開始!")
            
    def execute_training_round(self) -> Dict[str, Any]:
        """執行一輪聯邦學習"""
        self.current_round += 1
        round_info = {
            'round': self.current_round,
            'attack_active': self.attack_active,
            'timestamp': datetime.now().isoformat()
        }
        
        # 檢查是否應該啟動攻擊
        if not self.attack_active and self.current_round >= self.attack_start_round:
            self.attack_active = True
            print(f"🚨 Sybil 攻擊在第 {self.current_round} 輪開始!")
            
        global_model = self.environment.get_global_model()
        
        # 收集所有客戶端的本地模型更新
        local_models = []
        
        # 誠實客戶端訓練
        for client in self.environment.honest_clients:
            local_model = client.train_local_model(global_model)
            local_models.append(local_model)
            
        # Sybil 客戶端訓練（如果攻擊激活）
        if self.attack_active:
            for client in self.environment.sybil_clients:
                local_model = client.train_local_model(global_model)
                local_models.append(local_model)
                
        # 聯邦平均聚合
        updated_global_model = self.federated_averaging(local_models)
        self.environment.global_model = updated_global_model
        
        # 評估模型性能
        accuracy, loss = self.environment.evaluate_model(updated_global_model)
        
        # 記錄結果
        round_info.update({
            'accuracy': accuracy,
            'loss': loss,
            'num_participants': len(local_models),
            'sybil_ratio': len(self.environment.sybil_clients) / len(local_models) if self.attack_active else 0
        })
        
        self.attack_history.append(round_info)
        self.environment.training_history['rounds'].append(self.current_round)
        self.environment.training_history['accuracy'].append(accuracy)
        self.environment.training_history['loss'].append(loss)
        
        # 輸出進度
        status = "🔥 攻擊中" if self.attack_active else "🔐 正常"
        print(f"第 {self.current_round} 輪 | {status} | 準確率: {accuracy:.4f} | 損失: {loss:.4f}")
        
        return round_info
    
    def run_attack_simulation(self, total_rounds: int = 10, attack_start_round: int = 3,
                            verbose: bool = True) -> Dict[str, Any]:
        """運行完整的攻擊模擬"""
        
        if verbose:
            print("=" * 70)
            print("🎯 Sybil 攻擊模擬開始")
            print("=" * 70)
            env_info = self.environment.get_environment_info()
            print(f"📊 環境設置:")
            print(f"   誠實客戶端: {env_info['num_honest_clients']}")
            print(f"   Sybil客戶端: {env_info['num_sybil_clients']}")
            print(f"   Sybil比例: {env_info['sybil_ratio']:.2%}")
            print(f"   攻擊開始輪次: {attack_start_round}")
            print(f"   總輪次: {total_rounds}")
            print("-" * 70)
        
        # 設置攻擊開始輪次
        self.attack_start_round = attack_start_round
        
        # 執行訓練輪次
        for round_num in range(total_rounds):
            round_result = self.execute_training_round()
            
        # 計算攻擊效果
        attack_results = self.analyze_attack_effectiveness()
        
        if verbose:
            print("-" * 70)
            print("📈 攻擊結果分析:")
            print(f"   最大準確率下降: {attack_results['max_accuracy_drop']:.4f}")
            print(f"   平均攻擊影響: {attack_results['avg_attack_impact']:.4f}")
            print(f"   攻擊效果等級: {attack_results['effectiveness_level']}")
            print("=" * 70)
        
        return {
            'environment_info': self.environment.get_environment_info(),
            'attack_config': {
                'total_rounds': total_rounds,
                'attack_start_round': attack_start_round
            },
            'results': attack_results,
            'history': self.attack_history,
            'final_accuracy': self.attack_history[-1]['accuracy'] if self.attack_history else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_attack_effectiveness(self) -> Dict[str, Any]:
        """分析攻擊效果"""
        if not self.attack_history:
            return {'error': '沒有攻擊歷史記錄'}
            
        # 找到攻擊前後的準確率
        pre_attack_accuracy = []
        during_attack_accuracy = []
        
        for record in self.attack_history:
            if record['attack_active']:
                during_attack_accuracy.append(record['accuracy'])
            else:
                pre_attack_accuracy.append(record['accuracy'])
                
        if not pre_attack_accuracy or not during_attack_accuracy:
            return {'error': '攻擊數據不足，無法分析'}
            
        # 計算攻擊效果指標
        avg_pre_attack = np.mean(pre_attack_accuracy)
        avg_during_attack = np.mean(during_attack_accuracy)
        max_pre_attack = max(pre_attack_accuracy)
        min_during_attack = min(during_attack_accuracy)
        
        max_accuracy_drop = max_pre_attack - min_during_attack
        avg_attack_impact = avg_pre_attack - avg_during_attack
        
        # 評估攻擊效果等級
        if max_accuracy_drop > 0.1:
            effectiveness = "高效"
        elif max_accuracy_drop > 0.05:
            effectiveness = "中效"
        else:
            effectiveness = "低效"
            
        return {
            'max_accuracy_drop': max_accuracy_drop,
            'avg_attack_impact': avg_attack_impact,
            'avg_pre_attack_accuracy': avg_pre_attack,
            'avg_during_attack_accuracy': avg_during_attack,
            'effectiveness_level': effectiveness,
            'total_rounds': len(self.attack_history),
            'attack_rounds': len(during_attack_accuracy)
        }
    
    def save_results(self, filepath: str = "sybil_attack_results.json"):
        """保存攻擊結果到文件"""
        results = {
            'environment_info': self.environment.get_environment_info(),
            'attack_effectiveness': self.analyze_attack_effectiveness(),
            'training_history': self.attack_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"💾 結果已保存到: {filepath}")
        return filepath
    
    def visualize_attack_progress(self):
        """視覺化攻擊進度（簡化版本，使用文字圖表）"""
        if not self.attack_history:
            print("❌ 沒有攻擊歷史記錄可以顯示")
            return
            
        print("\n📊 攻擊進度可視化:")
        print("-" * 50)
        
        accuracies = [record['accuracy'] for record in self.attack_history]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        acc_range = max_acc - min_acc if max_acc != min_acc else 0.1
        
        for i, record in enumerate(self.attack_history):
            round_num = record['round']
            accuracy = record['accuracy']
            attack_status = "🔥" if record['attack_active'] else "🔐"
            
            # 創建簡單的條形圖
            if acc_range > 0:
                bar_length = int(40 * (accuracy - min_acc) / acc_range)
            else:
                bar_length = 20
                
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"R{round_num:2d} {attack_status} |{bar}| {accuracy:.4f}")
            
        print("-" * 50)
        print(f"範圍: {min_acc:.4f} ~ {max_acc:.4f}")
        print("圖例: 🔐=正常訓練, 🔥=攻擊中")

def create_attack_orchestrator(environment: FederatedLearningEnvironment) -> SybilAttackOrchestrator:
    """創建攻擊編排器"""
    return SybilAttackOrchestrator(environment)

# 預定義攻擊場景
ATTACK_SCENARIOS = {
    'mild': {
        'attack_start_round': 5,
        'total_rounds': 10,
        'description': '溫和攻擊 - 較晚開始，影響較小'
    },
    'moderate': {
        'attack_start_round': 3,
        'total_rounds': 12,
        'description': '中等攻擊 - 中期開始，平衡的影響'
    },
    'aggressive': {
        'attack_start_round': 2,
        'total_rounds': 15,
        'description': '激進攻擊 - 早期開始，持續時間長'
    },
    'stealth': {
        'attack_start_round': 8,
        'total_rounds': 15,
        'description': '隱蔽攻擊 - 很晚才開始，難以被發現'
    }
} 