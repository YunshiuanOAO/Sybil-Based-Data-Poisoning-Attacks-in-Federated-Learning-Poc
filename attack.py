"""
Sybil-based Virtual Data Poisoning Attack Module
===============================================

基於論文: "Sybil-based Virtual Data Poisoning Attacks in Federated Learning"
實現無目標 (Untargeted) 攻擊變體

核心功能：
- 虛擬數據生成 (基於梯度匹配)
- 目標模型獲取 (Online Global 方案)
- Sybil 節點管理
- 無目標投毒攻擊

Author: Security Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any
from environment import FederatedLearningEnvironment

class SybilVirtualDataAttackOrchestrator:
    """
    基於論文的 Sybil 虛擬數據投毒攻擊編排器
    實現無目標攻擊 (Untargeted Attack)
    """
    
    def __init__(self, environment: FederatedLearningEnvironment, num_sybil_per_malicious: int = 5):
        self.environment = environment
        self.attack_history = []
        self.current_round = 0
        self.attack_active = False
        self.attack_start_round = 0
        
        # 虛擬數據攻擊參數
        self.num_sybil_per_malicious = num_sybil_per_malicious  # 每個惡意客戶端的 sybil 節點數
        self.perturbation_lr = 0.1  # 擾動學習率
        self.max_perturbation_iters = 50  # 最大擾動迭代次數
        self.virtual_datasets = {}  # 存儲虛擬投毒數據
        self.target_model = None  # 目標模型
        
    def acquire_target_model_online_global(self, global_model: nn.Module) -> nn.Module:
        """
        目標模型獲取 - Online Global 方案 (修復版)
        創建一個故意性能較差的目標模型來指導攻擊方向
        """
        if len(self.environment.sybil_clients) == 0:
            return copy.deepcopy(global_model)
            
        # 創建一個故意降級的目標模型
        target_model = copy.deepcopy(global_model)
        target_model.train()
        
        # 使用高學習率和錯誤梯度方向來降級模型
        optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1)  # 高學習率
        criterion = nn.CrossEntropyLoss()
        
        # 收集一些數據來進行反向訓練
        collected_data = []
        collected_labels = []
        
        for client in self.environment.sybil_clients:
            for batch_idx, (data, target) in enumerate(client.data_loader):
                if batch_idx >= 3:  # 限制批次
                    break
                collected_data.append(data)
                collected_labels.append(target)
        
        if collected_data:
            all_data = torch.cat(collected_data, dim=0)
            all_labels = torch.cat(collected_labels, dim=0)
            
            # 執行"反向訓練"來降級模型
            for step in range(10):  # 多步降級
                optimizer.zero_grad()
                output = target_model(all_data)
                
                # 策略1: 使用反轉的標籤 (無目標攻擊的有效方式)
                flipped_labels = (all_labels + torch.randint(1, self.environment.num_classes, all_labels.shape)) % self.environment.num_classes
                loss = criterion(output, flipped_labels)
                
                # 策略2: 添加噪聲來破壞權重
                loss.backward()
                
                # 在更新前添加噪聲
                for param in target_model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * 0.01
                
                optimizer.step()
                    
        return target_model
    
    def gradient_matching_perturbation(self, base_data: torch.Tensor, base_labels: torch.Tensor, 
                                     global_model: nn.Module, target_model: nn.Module) -> torch.Tensor:
        """
        基於梯度匹配的虛擬數據生成 (修復版)
        創建能夠指向錯誤梯度方向的投毒數據
        """
        base_data = base_data.clone().detach().requires_grad_(True)
        perturbation = torch.zeros_like(base_data, requires_grad=True)
        
        # 計算破壞性目標梯度 (從差異較大的模型指向全局模型)
        target_gradient = []
        for global_param, target_param in zip(global_model.parameters(), target_model.parameters()):
            # 讓梯度指向目標模型的方向，這會破壞全局模型
            target_gradient.append((target_param.data - global_param.data).flatten())
        target_gradient = torch.cat(target_gradient)
        
        optimizer = torch.optim.Adam([perturbation], lr=self.perturbation_lr)
        
        for iteration in range(self.max_perturbation_iters):
            optimizer.zero_grad()
            
            # 對擾動後的數據計算梯度
            perturbed_data = base_data + perturbation
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            global_model.zero_grad()
            output = global_model(perturbed_data)
            
            # 無目標攻擊策略：使用最混淆的標籤配對
            worst_labels = self._generate_adversarial_labels(output, base_labels)
            loss = F.cross_entropy(output, worst_labels)
            
            # 計算關於模型參數的梯度
            model_gradients = torch.autograd.grad(loss, global_model.parameters(), 
                                                create_graph=True, retain_graph=True)
            poison_gradient = torch.cat([g.flatten() for g in model_gradients])
            
            # 讓投毒梯度與破壞性目標梯度對齊
            cosine_sim = F.cosine_similarity(target_gradient.unsqueeze(0), 
                                           poison_gradient.unsqueeze(0), dim=1)
            
            # 最大化相似度，使攻擊更有效
            objective = -cosine_sim  # 負號：最大化相似度
            
            objective.backward()
            optimizer.step()
            
            # 限制擾動幅度，但允許更大的擾動
            with torch.no_grad():
                perturbation.clamp_(-0.2, 0.2)
        
        return perturbation.detach()
    
    def _generate_adversarial_labels(self, output: torch.Tensor, original_labels: torch.Tensor) -> torch.Tensor:
        """
        生成最具對抗性的標籤
        選擇模型預測置信度最低的標籤作為目標
        """
        with torch.no_grad():
            # 獲取模型預測的概率分佈
            probs = F.softmax(output, dim=1)
            
            # 對每個樣本，選擇置信度最低的標籤（但不是原標籤）
            adversarial_labels = []
            
            for i in range(len(original_labels)):
                # 排除原始標籤
                masked_probs = probs[i].clone()
                masked_probs[original_labels[i]] = 1.0  # 排除原標籤
                
                # 選擇置信度最低的標籤
                worst_label = torch.argmin(masked_probs)
                adversarial_labels.append(worst_label)
            
            return torch.stack(adversarial_labels)
    
    def generate_virtual_poisoning_data(self, global_model: nn.Module) -> Dict[str, torch.utils.data.Dataset]:
        """
        為所有 Sybil 節點生成虛擬投毒數據
        論文算法 1 的核心部分
        """
        # 獲取目標模型
        self.target_model = self.acquire_target_model_online_global(global_model)
        
        virtual_datasets = {}
        
        # 為每個惡意客戶端的 sybil 節點生成虛擬數據
        for mal_idx, malicious_client in enumerate(self.environment.sybil_clients):
            # 從惡意客戶端採樣基線數據
            base_samples = []
            base_labels = []
            
            # 收集基線數據 (論文公式 6)
            sample_count = 0
            for data, label in malicious_client.data_loader:
                if sample_count >= 32:  # 限制樣本數量
                    break
                base_samples.append(data)
                base_labels.append(label)
                sample_count += data.size(0)
            
            if base_samples:
                base_data = torch.cat(base_samples, dim=0)
                base_labels = torch.cat(base_labels, dim=0)
                
                # 生成擾動
                perturbations = self.gradient_matching_perturbation(
                    base_data, base_labels, global_model, self.target_model
                )
                
                # 創建投毒數據
                poisoned_data = base_data + perturbations
                poisoned_data = torch.clamp(poisoned_data, 0, 1)
                
                # 為每個 sybil 節點創建虛擬數據集
                for sybil_idx in range(self.num_sybil_per_malicious):
                    dataset_key = f"malicious_{mal_idx}_sybil_{sybil_idx}"
                    
                    # 創建虛擬數據集
                    virtual_dataset = torch.utils.data.TensorDataset(
                        poisoned_data, 
                        base_labels  # 保持原始標籤用於混淆
                    )
                    virtual_datasets[dataset_key] = virtual_dataset
        
        self.virtual_datasets = virtual_datasets
        return virtual_datasets
        
    def federated_averaging_with_sybil(self, honest_models: List[nn.Module], 
                                      sybil_models: List[nn.Module]) -> nn.Module:
        """
        聊邦平均聚合包含 Sybil 節點 - 穩定版
        實現有效但穩定的權重重分配
        """
        if not honest_models and not sybil_models:
            return self.environment.global_model
            
        total_models = len(honest_models) + len(sybil_models)
        
        if len(sybil_models) == 0:
            # 沒有 Sybil 節點，正常聚合
            return self._normal_federated_averaging(honest_models)
        
        # 🔧 平衡的權重策略（避免過度激進）
        if len(sybil_models) > 0:
            # 給 Sybil 節點分配 70% 的權重（有效但不過度）
            sybil_total_weight = 0.70
            honest_total_weight = 0.30
            
            sybil_weight_per_model = sybil_total_weight / len(sybil_models)
            honest_weight_per_model = honest_total_weight / len(honest_models) if honest_models else 0
            
            print(f"   🎯 平衡權重重分配:")
            print(f"      誠實客戶端: {honest_weight_per_model:.4f} x {len(honest_models)} = {honest_total_weight:.1%}")
            print(f"      Sybil 節點: {sybil_weight_per_model:.4f} x {len(sybil_models)} = {sybil_total_weight:.1%}")
        else:
            sybil_weight_per_model = 0
            honest_weight_per_model = 1.0 / len(honest_models)
        
        # 創建全新的全局模型
        global_model = copy.deepcopy(self.environment.global_model)
        
        # 清零所有參數
        for param in global_model.parameters():
            param.data.zero_()
        
        # 聚合誠實客戶端
        for model in honest_models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                # 檢查參數有效性
                if not torch.isnan(local_param.data).any() and not torch.isinf(local_param.data).any():
                    global_param.data += honest_weight_per_model * local_param.data
        
        # 聚合 Sybil 節點（主導地位但穩定）
        for model in sybil_models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                # 檢查參數有效性
                if not torch.isnan(local_param.data).any() and not torch.isinf(local_param.data).any():
                    # 基礎權重貢獻
                    global_param.data += sybil_weight_per_model * local_param.data
        
        # 🔧 最終穩定性檢查
        with torch.no_grad():
            for param in global_model.parameters():
                # 檢查並修正異常值
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    print(f"⚠️ 全局模型檢測到異常參數，重置")
                    param.data.copy_(self.environment.global_model.state_dict()[list(self.environment.global_model.state_dict().keys())[0]])
                
                # 限制參數範圍防止爆炸
                param.data = torch.clamp(param.data, -5.0, 5.0)
                
                # 輕微的破壞性擾動（溫和版本）
                if torch.rand(1).item() < 0.2:  # 20% 概率，降低破壞性
                    # 隨機縮放參數（範圍更保守）
                    scale_factor = torch.rand(1).item() * (1.2 - 0.8) + 0.8  # 0.8 到 1.2 之間
                    param.data *= scale_factor
        
        return global_model
    
    def _normal_federated_averaging(self, models: List[nn.Module]) -> nn.Module:
        """正常的聯邦平均算法"""
        if not models:
            return self.environment.global_model
            
        global_model = copy.deepcopy(models[0])
        
        # 清零參數
        for param in global_model.parameters():
            param.data.zero_()
            
        # 等權重平均
        weight = 1.0 / len(models)
        for model in models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                global_param.data += weight * local_param.data
                
        return global_model
    
    def start_attack(self, start_round: int = 3):
        """開始 Sybil 攻擊"""
        self.attack_start_round = start_round
        if self.current_round >= start_round:
            self.attack_active = True
            print(f"🚨 Sybil 攻擊已在第 {self.current_round} 輪開始!")
            
    def train_sybil_virtual_nodes(self, global_model: nn.Module) -> List[nn.Module]:
        """
        使用虛擬數據訓練 Sybil 節點 (修復版)
        實現真正的破壞性訓練
        """
        sybil_models = []
        
        if not self.virtual_datasets:
            return sybil_models
            
        for dataset_key, virtual_dataset in self.virtual_datasets.items():
            # 為每個虛擬數據集訓練一個模型
            sybil_model = copy.deepcopy(global_model)
            sybil_model.train()
            
            # 使用更高的學習率進行破壞性訓練
            optimizer = torch.optim.SGD(sybil_model.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()
            
            # 創建 DataLoader
            data_loader = torch.utils.data.DataLoader(virtual_dataset, batch_size=16, shuffle=True)
            
            # 執行破壞性訓練
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 5:  # 增加訓練批次
                    break
                    
                optimizer.zero_grad()
                output = sybil_model(data)
                
                # 策略1: 使用最混淆的標籤
                adversarial_target = self._generate_adversarial_labels(output, target)
                loss = criterion(output, adversarial_target)
                
                loss.backward()
                
                # 策略2: 在梯度中添加破壞性噪聲
                with torch.no_grad():
                    for param in sybil_model.parameters():
                        if param.grad is not None:
                            # 添加針對性噪聲
                            noise = torch.randn_like(param.grad) * 0.02
                            param.grad += noise
                
                optimizer.step()
                
                # 策略3: 週期性地添加權重噪聲
                if batch_idx % 2 == 0:
                    with torch.no_grad():
                        for param in sybil_model.parameters():
                            param.data += torch.randn_like(param.data) * 0.001
                
            sybil_models.append(sybil_model)
            
        return sybil_models
    
    def analyze_attack_effectiveness(self) -> Dict[str, Any]:
        """分析攻擊效果 - 包含 SPoiL 風格評估指標"""
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
            
        # 計算基本攻擊效果指標
        avg_pre_attack = np.mean(pre_attack_accuracy)
        avg_during_attack = np.mean(during_attack_accuracy)
        max_pre_attack = max(pre_attack_accuracy)
        min_during_attack = min(during_attack_accuracy)
        
        max_accuracy_drop = max_pre_attack - min_during_attack
        avg_attack_impact = avg_pre_attack - avg_during_attack
        
        # 🆕 SPoiL 風格評估指標
        
        # 1. Main Task Accuracy (MTA) - 主任務準確率
        # 最終輪次的準確率作為主任務準確率
        main_task_accuracy = self.attack_history[-1]['accuracy'] if self.attack_history else 0
        
        # 2. Poisoning Success Rate (PSR) - 投毒成功率
        # 計算攻擊期間準確率持續下降的比例
        poisoning_success_count = 0
        baseline_accuracy = max_pre_attack if pre_attack_accuracy else 0.5
        
        for record in self.attack_history:
            if record['attack_active'] and record['accuracy'] < baseline_accuracy:
                poisoning_success_count += 1
                
        poisoning_success_rate = (poisoning_success_count / len(during_attack_accuracy)) if during_attack_accuracy else 0
        
        # 3. Attack Persistence - 攻擊持續性
        # 計算攻擊效果是否在攻擊期間保持
        attack_persistence = 0
        if len(during_attack_accuracy) > 1:
            consistent_degradation = 0
            for i in range(1, len(during_attack_accuracy)):
                if during_attack_accuracy[i] <= during_attack_accuracy[i-1] * 1.02:  # 允許2%的波動
                    consistent_degradation += 1
            attack_persistence = consistent_degradation / (len(during_attack_accuracy) - 1)
        
        # 4. 相對性能下降 (Relative Performance Degradation)
        relative_degradation = (avg_pre_attack - avg_during_attack) / avg_pre_attack if avg_pre_attack > 0 else 0
        
        # 評估攻擊效果等級（基於多個指標）
        if max_accuracy_drop > 0.1 and poisoning_success_rate > 0.8:
            effectiveness = "高效"
        elif max_accuracy_drop > 0.05 and poisoning_success_rate > 0.6:
            effectiveness = "中效"
        else:
            effectiveness = "低效"
            
        return {
            # 原有指標
            'max_accuracy_drop': max_accuracy_drop,
            'avg_attack_impact': avg_attack_impact,
            'avg_pre_attack_accuracy': avg_pre_attack,
            'avg_during_attack_accuracy': avg_during_attack,
            'effectiveness_level': effectiveness,
            'total_rounds': len(self.attack_history),
            'attack_rounds': len(during_attack_accuracy),
            
            # 🆕 SPoiL 風格指標
            'main_task_accuracy': main_task_accuracy,
            'poisoning_success_rate': poisoning_success_rate,
            'attack_persistence': attack_persistence,
            'relative_performance_degradation': relative_degradation,
            
            # 詳細分析
            'baseline_accuracy': max_pre_attack,
            'final_accuracy': main_task_accuracy,
            'accuracy_degradation_percentage': relative_degradation * 100,
            'attack_sustainability': attack_persistence > 0.7
        }
    
    def simple_label_flipping_attack(self, global_model: nn.Module, flip_ratio: float = 0.3) -> List[nn.Module]:
        """
        🆕 實現穩定的 SPoiL 風格攻擊 - 修復版
        
        Args:
            global_model: 全局模型
            flip_ratio: 標籤翻轉比例
            
        Returns:
            List[nn.Module]: 訓練後的 Sybil 模型列表
        """
        sybil_models = []
        
        if len(self.environment.sybil_clients) == 0:
            return sybil_models
            
        # 為每個惡意客戶端創建多個 Sybil 節點
        for mal_idx, malicious_client in enumerate(self.environment.sybil_clients):
            # 收集原始數據
            original_data = []
            original_labels = []
            
            for data, labels in malicious_client.data_loader:
                original_data.append(data)
                original_labels.append(labels)
                
            if original_data:
                all_data = torch.cat(original_data, dim=0)
                all_labels = torch.cat(original_labels, dim=0)
                
                # 🔧 穩定的破壞性策略
                
                # 策略1: 標籤翻轉（根據 flip_ratio）
                num_to_flip = int(len(all_labels) * flip_ratio)
                flip_indices = torch.randperm(len(all_labels))[:num_to_flip]
                flipped_labels = all_labels.clone()
                
                # 使用智能標籤翻轉：翻轉到最遠的類別
                for idx in flip_indices:
                    original_class = all_labels[idx].item()
                    # 翻轉到最遠的類別（對於 MNIST，0->9, 1->8, 等等）
                    flipped_class = (self.environment.num_classes - 1) - original_class
                    flipped_labels[idx] = flipped_class
                
                # 策略2: 輕微的對抗性噪聲（避免過度破壞）
                adversarial_data = all_data.clone()
                if torch.rand(1).item() < 0.5:  # 50% 概率添加噪聲
                    noise_strength = 0.1  # 降低噪聲強度
                    noise = torch.randn_like(all_data) * noise_strength
                    adversarial_data = torch.clamp(all_data + noise, 0, 1)
                
                # 為每個 Sybil 節點創建模型
                for sybil_idx in range(self.num_sybil_per_malicious):
                    sybil_model = copy.deepcopy(global_model)
                    sybil_model.train()
                    
                    # 🔧 使用適度的學習率
                    optimizer = torch.optim.SGD(sybil_model.parameters(), lr=0.01, momentum=0.5)
                    criterion = nn.CrossEntropyLoss()
                    
                    # 創建破壞性數據集
                    poison_dataset = torch.utils.data.TensorDataset(adversarial_data, flipped_labels)
                    data_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=32, shuffle=True)
                    
                    # 🆕 穩定的訓練策略
                    for epoch in range(3):  # 減少訓練輪數
                        for batch_idx, (data, target) in enumerate(data_loader):
                            if batch_idx >= 5:  # 限制批次數
                                break
                                
                            optimizer.zero_grad()
                            output = sybil_model(data)
                            
                            # 使用標準損失函數，但訓練錯誤標籤
                            loss = criterion(output, target)
                            
                            # 檢查損失是否有效
                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f"⚠️ 檢測到無效損失，跳過此批次")
                                continue
                            
                            loss.backward()
                            
                            # 🔧 梯度裁剪防止爆炸
                            torch.nn.utils.clip_grad_norm_(sybil_model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            
                            # 🆕 溫和的參數擾動
                            if epoch == 2 and batch_idx == 4:  # 只在最後一次添加擾動
                                with torch.no_grad():
                                    for param in sybil_model.parameters():
                                        # 添加小幅度隨機擾動
                                        noise = torch.randn_like(param) * 0.01
                                        param.data += noise
                    
                    # 🔧 最終參數檢查和修正
                    with torch.no_grad():
                        for param in sybil_model.parameters():
                            # 檢查並修正異常值
                            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                                print(f"⚠️ 檢測到異常參數，重置為全局模型參數")
                                param.data.copy_(global_model.state_dict()[list(global_model.state_dict().keys())[0]])
                            
                            # 限制參數範圍
                            param.data = torch.clamp(param.data, -10.0, 10.0)
                    
                    sybil_models.append(sybil_model)
        
        return sybil_models
            
    def execute_training_round(self) -> Dict[str, Any]:
        """
        執行一輪訓練 - 支持多種攻擊方法
        """
        self.current_round += 1
        round_info = {
            'round': self.current_round,
            'attack_active': self.attack_active,
            'timestamp': datetime.now().isoformat(),
            'num_sybil_nodes': 0,
            'virtual_data_generated': False,
            'attack_method': 'none'
        }
        
        # 檢查是否應該啟動攻擊
        if not self.attack_active and self.current_round >= self.attack_start_round:
            self.attack_active = True
            print(f"🚨 Sybil 攻擊在第 {self.current_round} 輪開始!")
            
        global_model = self.environment.get_global_model()
        
        # 誠實客戶端訓練
        honest_models = []
        for client in self.environment.honest_clients:
            local_model = client.train_local_model(global_model)
            honest_models.append(local_model)
            
        # Sybil 攻擊執行
        sybil_models = []
        if self.attack_active:
            # 🆕 支持多種攻擊方法
            attack_method = getattr(self, 'attack_method', 'virtual_data')
            
            if attack_method == 'label_flipping':
                # SPoiL 風格的標籤翻轉攻擊
                scenario_config = ATTACK_SCENARIOS.get(getattr(self, 'current_scenario', 'spoil_replica'), {})
                flip_ratio = scenario_config.get('flip_ratio', 0.3)
                sybil_models = self.simple_label_flipping_attack(global_model, flip_ratio=flip_ratio)
                round_info['attack_method'] = 'label_flipping'
                round_info['flip_ratio'] = flip_ratio
                print(f"   📊 執行標籤翻轉攻擊，翻轉比例: {flip_ratio:.1%}，創建了 {len(sybil_models)} 個 Sybil 節點")
                
            else:
                # 原有的虛擬數據攻擊
                self.generate_virtual_poisoning_data(global_model)
                sybil_models = self.train_sybil_virtual_nodes(global_model)
                round_info['virtual_data_generated'] = True
                round_info['attack_method'] = 'virtual_data'
                print(f"   📊 生成了 {len(self.virtual_datasets)} 個虛擬數據集，訓練了 {len(sybil_models)} 個 Sybil 節點")
            
            round_info['num_sybil_nodes'] = len(sybil_models)
                
        # 聯邦平均聚合（包含 Sybil 節點）
        updated_global_model = self.federated_averaging_with_sybil(honest_models, sybil_models)
        self.environment.global_model = updated_global_model
        
        # 評估模型性能
        accuracy, loss = self.environment.evaluate_model(updated_global_model)
        
        # 記錄攻擊歷史
        round_info.update({
            'accuracy': accuracy,
            'loss': loss,
            'num_participants': len(honest_models) + len(sybil_models),
            'honest_clients': len(honest_models),
            'sybil_nodes': len(sybil_models),
            'sybil_ratio': len(sybil_models) / (len(honest_models) + len(sybil_models)) if (len(honest_models) + len(sybil_models)) > 0 else 0
        })
        
        self.attack_history.append(round_info)
        
        # 顯示進度
        if len(sybil_models) > 0:
            if round_info['attack_method'] == 'label_flipping':
                status = f"🎯 標籤翻轉攻擊 (Sybil: {len(sybil_models)})"
            else:
                status = f"🔥 虛擬攻擊中 (Sybil: {len(sybil_models)})"
        else:
            status = "🔐 正常"
        print(f"第 {self.current_round} 輪 | {status} | 準確率: {accuracy:.4f} | 損失: {loss:.4f}")
        
        return round_info
    
    def set_attack_method(self, method: str):
        """
        🆕 設置攻擊方法
        
        Args:
            method: 'virtual_data' 或 'label_flipping'
        """
        if method in ['virtual_data', 'label_flipping']:
            self.attack_method = method
        else:
            raise ValueError("攻擊方法必須是 'virtual_data' 或 'label_flipping'")
    
    def set_current_scenario(self, scenario_name: str):
        """
        🆕 設置當前攻擊場景
        
        Args:
            scenario_name: 場景名稱
        """
        self.current_scenario = scenario_name
    
    def run_attack_simulation(self, total_rounds: int = 10, attack_start_round: int = 3,
                            verbose: bool = True, attack_method: str = 'virtual_data') -> Dict[str, Any]:
        """運行完整的攻擊模擬 - 支持多種攻擊方法"""
        
        # 🆕 設置攻擊方法
        self.set_attack_method(attack_method)
        
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
            print(f"   攻擊方法: {attack_method}")
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
            print(f"   🆕 主任務準確率 (MTA): {attack_results['main_task_accuracy']:.4f}")
            print(f"   🆕 投毒成功率 (PSR): {attack_results['poisoning_success_rate']:.2%}")
            print(f"   🆕 攻擊持續性: {attack_results['attack_persistence']:.2%}")
            print(f"   🆕 相對性能下降: {attack_results['relative_performance_degradation']:.2%}")
            print(f"   攻擊效果等級: {attack_results['effectiveness_level']}")
            print("=" * 70)
        
        return {
            'environment_info': self.environment.get_environment_info(),
            'attack_config': {
                'total_rounds': total_rounds,
                'attack_start_round': attack_start_round,
                'attack_method': attack_method
            },
            'results': attack_results,
            'history': self.attack_history,
            'final_accuracy': self.attack_history[-1]['accuracy'] if self.attack_history else 0,
            'timestamp': datetime.now().isoformat()
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

def create_attack_orchestrator(environment: FederatedLearningEnvironment, 
                              num_sybil_per_malicious: int = 5) -> SybilVirtualDataAttackOrchestrator:
    """
    創建基於論文的虛擬數據攻擊編排器
    
    Args:
        environment: 聯邦學習環境
        num_sybil_per_malicious: 每個惡意客戶端生成的 Sybil 節點數量
    """
    return SybilVirtualDataAttackOrchestrator(environment, num_sybil_per_malicious)

# 基於論文的虛擬數據攻擊場景
VIRTUAL_DATA_ATTACK_SCENARIOS = {
    'mild_virtual': {
        'attack_start_round': 5,
        'total_rounds': 10,
        'num_sybil_per_malicious': 3,
        'attack_method': 'virtual_data',
        'description': '溫和虛擬攻擊 - 較晚開始，少量 Sybil 節點'
    },
    'moderate_virtual': {
        'attack_start_round': 3,
        'total_rounds': 12,
        'num_sybil_per_malicious': 5,
        'attack_method': 'virtual_data',
        'description': '中等虛擬攻擊 - 中期開始，平衡的 Sybil 節點數量'
    },
    'aggressive_virtual': {
        'attack_start_round': 2,
        'total_rounds': 15,
        'num_sybil_per_malicious': 8,
        'attack_method': 'virtual_data',
        'description': '激進虛擬攻擊 - 早期開始，大量 Sybil 節點'
    },
    'stealth_virtual': {
        'attack_start_round': 8,
        'total_rounds': 15,
        'num_sybil_per_malicious': 4,
        'attack_method': 'virtual_data',
        'description': '隱蔽虛擬攻擊 - 很晚才開始，難以被發現'
    },
    'paper_replica': {
        'attack_start_round': 3,
        'total_rounds': 15,
        'num_sybil_per_malicious': 5,
        'attack_method': 'virtual_data',
        'description': '論文複現 - 按照虛擬數據論文設置的攻擊'
    },
    
    # 🆕 SPoiL 風格的標籤翻轉攻擊場景
    'spoil_mild': {
        'attack_start_round': 5,
        'total_rounds': 10,
        'num_sybil_per_malicious': 3,
        'attack_method': 'label_flipping',
        'flip_ratio': 0.2,
        'description': '溫和 SPoiL 攻擊 - 低翻轉比例，少量 Sybil'
    },
    'spoil_moderate': {
        'attack_start_round': 3,
        'total_rounds': 12,
        'num_sybil_per_malicious': 5,
        'attack_method': 'label_flipping',
        'flip_ratio': 0.3,
        'description': '中等 SPoiL 攻擊 - 中等翻轉比例，平衡設置'
    },
    'spoil_aggressive': {
        'attack_start_round': 2,
        'total_rounds': 15,
        'num_sybil_per_malicious': 8,
        'attack_method': 'label_flipping',
        'flip_ratio': 0.5,
        'description': '激進 SPoiL 攻擊 - 高翻轉比例，大量 Sybil'
    },
    'spoil_replica': {
        'attack_start_round': 3,  # 🔧 修正：改為第3輪開始攻擊，便於測試
        'total_rounds': 10,       # 🔧 修正：減少總輪數
        'num_sybil_per_malicious': 4,  # 每個惡意用戶創建4個 Sybil
        'attack_method': 'label_flipping',
        'flip_ratio': 0.3,
        'description': 'SPoiL 論文複現 - 調整為便於測試的設置'
    },
    'spoil_original': {
        'attack_start_round': 9,  # 對應 SPoiL 論文的第9輪開始攻擊
        'total_rounds': 20,
        'num_sybil_per_malicious': 4,  # 每個惡意用戶創建4個 Sybil
        'attack_method': 'label_flipping',
        'flip_ratio': 0.3,
        'description': 'SPoiL 論文原始設置 - 完全按照 2023 SPoiL 論文'
    }
}

# 保持向後兼容性的舊場景
ATTACK_SCENARIOS = VIRTUAL_DATA_ATTACK_SCENARIOS.copy()
ATTACK_SCENARIOS.update({
    'mild': VIRTUAL_DATA_ATTACK_SCENARIOS['mild_virtual'],
    'moderate': VIRTUAL_DATA_ATTACK_SCENARIOS['moderate_virtual'],
    'aggressive': VIRTUAL_DATA_ATTACK_SCENARIOS['aggressive_virtual'],
    'stealth': VIRTUAL_DATA_ATTACK_SCENARIOS['stealth_virtual']
}) 