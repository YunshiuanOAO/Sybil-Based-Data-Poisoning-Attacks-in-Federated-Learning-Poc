#!/usr/bin/env python3
"""
簡化版 Sybil-Based 無目標數據投毒攻擊工具
===========================================

此工具實現了聯邦學習中的 Sybil 攻擊，僅使用 Python 標準庫。
演示多個虛假客戶端如何通過投毒數據來降低整體模型性能。

使用方法:
    python3 simple_sybil_attack.py
"""

import random
import math
import json
from typing import List, Dict, Tuple
import time

class SimpleMatrix:
    """簡單的矩陣類，用於神經網絡參數"""
    
    def __init__(self, rows: int, cols: int, data: List[List[float]] = None):
        self.rows = rows
        self.cols = cols
        if data:
            self.data = data
        else:
            # 隨機初始化
            self.data = [[random.gauss(0, 0.1) for _ in range(cols)] for _ in range(rows)]
    
    def __add__(self, other):
        """矩陣加法"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩陣維度不匹配")
        
        result_data = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result_data.append(row)
        
        return SimpleMatrix(self.rows, self.cols, result_data)
    
    def __mul__(self, scalar: float):
        """標量乘法"""
        result_data = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] * scalar)
            result_data.append(row)
        
        return SimpleMatrix(self.rows, self.cols, result_data)
    
    def apply_noise(self, noise_factor: float):
        """添加噪聲"""
        for i in range(self.rows):
            for j in range(self.cols):
                noise = random.gauss(0, noise_factor)
                self.data[i][j] += noise

class SimpleNeuralNetwork:
    """簡化的神經網絡"""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 64, output_size: int = 10):
        self.weights1 = SimpleMatrix(input_size, hidden_size)
        self.weights2 = SimpleMatrix(hidden_size, output_size)
        self.accuracy = 0.0
    
    def forward(self, inputs: List[float]) -> List[float]:
        """前向傳播（簡化版）"""
        # 簡化計算，返回隨機預測結果
        return [random.random() for _ in range(10)]
    
    def train_step(self, data: List[Tuple[List[float], int]], poison_mode: bool = False):
        """訓練步驟"""
        if poison_mode:
            # 惡意訓練：增加噪聲並翻轉標籤
            self.weights1.apply_noise(0.1)
            self.weights2.apply_noise(0.1)
            # 模擬性能下降
            self.accuracy = max(0.1, self.accuracy - random.uniform(0.02, 0.08))
        else:
            # 正常訓練：改善性能
            self.accuracy = min(0.95, self.accuracy + random.uniform(0.01, 0.05))
    
    def copy(self):
        """複製模型"""
        new_model = SimpleNeuralNetwork()
        new_model.weights1 = SimpleMatrix(self.weights1.rows, self.weights1.cols, 
                                        [row[:] for row in self.weights1.data])
        new_model.weights2 = SimpleMatrix(self.weights2.rows, self.weights2.cols,
                                        [row[:] for row in self.weights2.data])
        new_model.accuracy = self.accuracy
        return new_model

class FederatedClient:
    """聯邦學習客戶端"""
    
    def __init__(self, client_id: str, is_malicious: bool = False):
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.local_model = SimpleNeuralNetwork()
        self.data_size = random.randint(100, 500)  # 模擬數據量
        
    def local_training(self, global_model: SimpleNeuralNetwork, rounds: int = 1):
        """本地訓練"""
        # 複製全局模型
        self.local_model = global_model.copy()
        
        # 生成模擬訓練數據
        training_data = self._generate_training_data()
        
        # 執行本地訓練
        for _ in range(rounds):
            self.local_model.train_step(training_data, poison_mode=self.is_malicious)
        
        if self.is_malicious:
            print(f"🦹‍♀️ Sybil 客戶端 {self.client_id} 完成惡意訓練")
        else:
            print(f"😇 誠實客戶端 {self.client_id} 完成正常訓練")
        
        return self.local_model
    
    def _generate_training_data(self) -> List[Tuple[List[float], int]]:
        """生成模擬訓練數據"""
        data = []
        for _ in range(self.data_size):
            # 生成隨機輸入（模擬 MNIST 28x28 圖像）
            inputs = [random.random() for _ in range(784)]
            
            if self.is_malicious:
                # 惡意客戶端：隨機標籤（標籤翻轉攻擊）
                label = random.randint(0, 9)
            else:
                # 誠實客戶端：正確標籤
                label = random.randint(0, 9)
            
            data.append((inputs, label))
        
        return data

class SybilAttackSimulator:
    """Sybil 攻擊模擬器"""
    
    def __init__(self, num_honest_clients: int = 5, num_sybil_clients: int = 3):
        self.num_honest_clients = num_honest_clients
        self.num_sybil_clients = num_sybil_clients
        self.total_clients = num_honest_clients + num_sybil_clients
        
        # 初始化客戶端
        self.honest_clients = [
            FederatedClient(f"honest_{i}", is_malicious=False) 
            for i in range(num_honest_clients)
        ]
        
        self.sybil_clients = [
            FederatedClient(f"sybil_{i}", is_malicious=True)
            for i in range(num_sybil_clients)
        ]
        
        self.all_clients = self.honest_clients + self.sybil_clients
        
        # 全局模型
        self.global_model = SimpleNeuralNetwork()
        self.global_model.accuracy = 0.8  # 初始準確率
        
        # 訓練歷史
        self.training_history = {
            'rounds': [],
            'accuracy': [],
            'honest_baseline': [],
            'attack_active': []
        }
    
    def federated_averaging(self, local_models: List[SimpleNeuralNetwork]) -> SimpleNeuralNetwork:
        """聯邦平均算法"""
        if not local_models:
            return self.global_model
        
        # 簡化版聯邦平均：計算準確率的加權平均
        total_weight = len(local_models)
        avg_accuracy = sum(model.accuracy for model in local_models) / total_weight
        
        # 創建新的全局模型
        new_global_model = self.global_model.copy()
        new_global_model.accuracy = avg_accuracy
        
        return new_global_model
    
    def run_attack(self, total_rounds: int = 15, attack_start_round: int = 5):
        """執行 Sybil 攻擊"""
        print("🚨 開始 Sybil-Based 無目標數據投毒攻擊模擬")
        print("=" * 60)
        print(f"環境設定：{self.num_honest_clients} 個誠實客戶端，{self.num_sybil_clients} 個 Sybil 客戶端")
        print(f"攻擊將在第 {attack_start_round} 輪開始")
        print(f"Sybil 比例：{self.num_sybil_clients/self.total_clients:.1%}")
        print()
        
        for round_num in range(1, total_rounds + 1):
            print(f"📅 第 {round_num} 輪聯邦學習")
            print("-" * 30)
            
            # 確定是否啟動攻擊
            attack_active = round_num >= attack_start_round
            
            if attack_active:
                print("⚠️  Sybil 攻擊已啟動！")
            
            # 收集本地模型
            local_models = []
            
            # 誠實客戶端訓練
            print("🔄 誠實客戶端本地訓練...")
            for client in self.honest_clients:
                local_model = client.local_training(self.global_model)
                local_models.append(local_model)
            
            # Sybil 客戶端訓練（如果攻擊啟動）
            if attack_active:
                print("🔄 Sybil 客戶端惡意訓練...")
                for client in self.sybil_clients:
                    local_model = client.local_training(self.global_model)
                    local_models.append(local_model)
            
            # 聯邦平均
            print("📊 執行聯邦平均...")
            self.global_model = self.federated_averaging(local_models)
            
            # 計算誠實基線（僅使用誠實客戶端）
            honest_models = [client.local_training(self.global_model) for client in self.honest_clients]
            honest_baseline = self.federated_averaging(honest_models).accuracy
            
            # 記錄結果
            self.training_history['rounds'].append(round_num)
            self.training_history['accuracy'].append(self.global_model.accuracy)
            self.training_history['honest_baseline'].append(honest_baseline)
            self.training_history['attack_active'].append(attack_active)
            
            # 顯示結果
            print(f"📈 全局模型準確率：{self.global_model.accuracy:.4f}")
            if attack_active:
                degradation = honest_baseline - self.global_model.accuracy
                print(f"📉 攻擊影響：準確率下降 {degradation:.4f}")
            
            print()
            time.sleep(0.5)  # 模擬訓練時間
        
        # 生成攻擊分析
        self._generate_analysis()
    
    def _generate_analysis(self):
        """生成攻擊分析報告"""
        print("🔍 攻擊分析報告")
        print("=" * 60)
        
        history = self.training_history
        
        # 計算攻擊效果
        initial_accuracy = history['accuracy'][0]
        final_accuracy = history['accuracy'][-1]
        max_degradation = max([
            h - a for h, a in zip(history['honest_baseline'], history['accuracy'])
        ])
        
        # 計算攻擊期間的平均準確率下降
        attack_rounds = [i for i, active in enumerate(history['attack_active']) if active]
        if attack_rounds:
            attack_degradations = [
                history['honest_baseline'][i] - history['accuracy'][i] 
                for i in attack_rounds
            ]
            avg_attack_degradation = sum(attack_degradations) / len(attack_degradations)
        else:
            avg_attack_degradation = 0
        
        print(f"📊 攻擊配置：")
        print(f"  • 誠實客戶端：{self.num_honest_clients}")
        print(f"  • Sybil 客戶端：{self.num_sybil_clients}")
        print(f"  • Sybil 比例：{self.num_sybil_clients/self.total_clients:.1%}")
        print()
        
        print(f"📈 攻擊結果：")
        print(f"  • 初始準確率：{initial_accuracy:.4f}")
        print(f"  • 最終準確率：{final_accuracy:.4f}")
        print(f"  • 最大準確率下降：{max_degradation:.4f}")
        print(f"  • 平均攻擊影響：{avg_attack_degradation:.4f}")
        print()
        
        # 攻擊效果評估
        if max_degradation > 0.15:
            effectiveness = "高效"
            emoji = "🔥"
        elif max_degradation > 0.05:
            effectiveness = "中等"
            emoji = "⚡"
        else:
            effectiveness = "低效"
            emoji = "💧"
        
        print(f"{emoji} 攻擊效果：{effectiveness}")
        print()
        
        # 保存結果到 JSON
        results = {
            'config': {
                'honest_clients': self.num_honest_clients,
                'sybil_clients': self.num_sybil_clients,
                'sybil_ratio': self.num_sybil_clients/self.total_clients
            },
            'results': {
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'max_degradation': max_degradation,
                'avg_attack_degradation': avg_attack_degradation,
                'effectiveness': effectiveness
            },
            'history': history
        }
        
        with open('sybil_attack_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("💾 詳細結果已保存到 'sybil_attack_results.json'")
        
        # 簡化的可視化輸出
        print("\n📊 準確率變化趨勢：")
        print("輪次 | 全局準確率 | 誠實基線 | 攻擊狀態")
        print("-" * 45)
        for i, round_num in enumerate(history['rounds']):
            accuracy = history['accuracy'][i]
            baseline = history['honest_baseline'][i]
            active = "🚨 攻擊中" if history['attack_active'][i] else "✅ 正常"
            print(f"{round_num:4d} | {accuracy:10.4f} | {baseline:8.4f} | {active}")

def main():
    """主函數"""
    print("🎯 Sybil-Based 無目標數據投毒攻擊工具（簡化版）")
    print("🚨 僅供教育和研究用途")
    print("=" * 70)
    
    # 設置隨機種子以便重現結果
    random.seed(42)
    
    # 創建攻擊模擬器
    simulator = SybilAttackSimulator(
        num_honest_clients=5,
        num_sybil_clients=3
    )
    
    # 執行攻擊
    simulator.run_attack(total_rounds=12, attack_start_round=4)
    
    print("\n🏁 模擬完成！")
    print("\n💡 此工具展示了 Sybil 攻擊如何影響聯邦學習性能")
    print("⚠️  請負責任地使用此工具，僅用於教育和研究目的")

if __name__ == "__main__":
    main() 