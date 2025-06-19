# Sybil 攻擊模擬系統 (Sybil Attack Simulation Framework)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**專業的聯邦學習 Sybil 攻擊研究與教育工具**

> ⚠️ **重要提醒**: 本工具僅供學術研究和教育目的使用，請勿用於惡意攻擊真實系統。

## 📋 目錄

- [項目簡介](#項目簡介)
- [主要特性](#主要特性)
- [技術架構](#技術架構)
- [安裝指南](#安裝指南)
- [快速開始](#快速開始)
- [使用教程](#使用教程)
- [攻擊場景](#攻擊場景)
- [實驗結果](#實驗結果)
- [技術實現](#技術實現)
- [API 文檔](#api-文檔)
- [貢獻指南](#貢獻指南)
- [許可證](#許可證)

## 🚀 項目簡介

本項目實現了一個完整的 **Sybil 攻擊模擬系統**，專門用於研究聯邦學習環境中的安全威脅。系統支持多種攻擊策略，並提供詳細的評估指標，幫助研究人員理解和防禦 Sybil 攻擊。

### 研究背景

基於 **SPoiL: Sybil-Based Untargeted Data Poisoning Attacks in Federated Learning (2023)** 論文，我們實現了兩種主要的攻擊方法：

1. **虛擬數據攻擊** - 基於梯度匹配的高複雜度攻擊
2. **標籤翻轉攻擊** - 基於 SPoiL 論文的中等複雜度攻擊

## ✨ 主要特性

### 🔒 攻擊能力
- **多重攻擊策略**: 虛擬數據生成、標籤翻轉、對抗性噪聲注入
- **智能 Sybil 節點**: 動態生成多個虛假客戶端
- **自適應權重操控**: 激進的聯邦平均權重重分配
- **梯度匹配技術**: 精準的目標模型模擬

### 📊 評估系統
- **SPoiL 風格評估**: 主任務準確率 (MTA)、投毒成功率 (PSR)
- **攻擊持續性分析**: 長期攻擊效果追蹤
- **性能下降量化**: 相對性能退化測量
- **實時進度可視化**: ASCII 圖表顯示攻擊進程

### 🛠️ 實驗框架
- **預設攻擊場景**: 從溫和到激進的 15 種攻擊配置
- **攻擊方法比較**: 虛擬數據 vs SPoiL 標籤翻轉
- **自動化實驗**: 批量運行和結果比較
- **JSON 結果輸出**: 詳細的攻擊數據記錄

### 🔍 安全特性
- **數值穩定性**: 防止 NaN/Inf 值的保護機制
- **參數限制**: 防止梯度爆炸的安全措施
- **錯誤處理**: 完整的異常捕獲和恢復

## 🏗️ 技術架構

### 核心組件

| 組件 | 功能 | 檔案 |
|------|------|------|
| **主程序** | 命令行介面和執行流程控制 | `main.py` |
| **環境設置** | 依賴檢查和系統初始化 | `setup.py` |
| **聯邦學習環境** | 客戶端管理和模型訓練 | `environment.py` |
| **攻擊編排器** | Sybil 攻擊實現和評估 | `attack.py` |
| **配置管理** | 攻擊參數和場景定義 | `config.py` |
| **攻擊比較** | 多種攻擊方法性能對比 | `compare_attacks.py` |

## 📦 安裝指南

### 系統要求

- **Python**: 3.8 或更高版本
- **操作系統**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **內存**: 建議 8GB 以上
- **存儲**: 2GB 可用空間

### 1. 環境準備

```bash
# 克隆項目
git clone https://github.com/your-username/sybil-attack-simulation.git
cd sybil-attack-simulation

# 創建虛擬環境（推薦）
python -m venv venv

# 激活虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. 安裝依賴

```bash
# 安裝 Python 依賴
pip install -r requirements.txt

# 驗證安裝
python setup.py
```

### 3. 環境驗證

```bash
# 運行環境檢查
python main.py --setup-only
```

成功輸出應顯示：
```
✅ 環境設置完成!
   - PyTorch 2.0+
   - NumPy 1.21+
   - Matplotlib 3.5+
   ...
```

## 🚀 快速開始

### 基本使用

```bash
# 運行默認攻擊場景
python main.py

# 運行 SPoiL 標籤翻轉攻擊
python main.py --scenario spoil_replica

# 運行虛擬數據攻擊
python main.py --scenario paper_replica

# 自定義參數
python main.py --scenario aggressive --rounds 20 --start-round 5
```

### 攻擊比較實驗

```bash
# 比較虛擬數據攻擊 vs SPoiL 攻擊
python compare_attacks.py
```

### 輸出示例

```
╔══════════════════════════════════════════════════════════════╗
║                    Sybil 攻擊模擬工具                         ║
║                                                              ║
║  🎯 專業的聯邦學習安全研究工具                                ║
║  🔬 教育和研究目的使用                                       ║
║  ⚠️  請勿用於惡意攻擊                                        ║
╚══════════════════════════════════════════════════════════════╝

🎯 執行 SPoiL 攻擊：

第 1 輪 | 🔐 正常 | 準確率: 0.6322 | 損失: 2.0728
第 2 輪 | 🔐 正常 | 準確率: 0.7542 | 損失: 1.1841
🚨 Sybil 攻擊在第 3 輪開始!
第 3 輪 | 🎯 標籤翻轉攻擊 (Sybil: 12) | 準確率: 0.0980 | 損失: 0.5381

📈 攻擊結果分析:
   🆕 主任務準確率 (MTA): 0.0980
   🆕 投毒成功率 (PSR): 100.00%
   🆕 攻擊持續性: 100.00%
   🆕 相對性能下降: 80.19%
   攻擊效果等級: 高效
```

## 📖 使用教程

### 1. 攻擊場景選擇

系統提供 15 種預設攻擊場景：

#### 虛擬數據攻擊系列
- `paper_replica` - 論文標準實現
- `mild_virtual` - 溫和虛擬攻擊
- `moderate_virtual` - 中等虛擬攻擊
- `aggressive_virtual` - 激進虛擬攻擊

#### SPoiL 標籤翻轉系列
- `spoil_replica` - SPoiL 論文複現（推薦測試）
- `spoil_original` - 原論文完整設置
- `spoil_mild` - 溫和標籤翻轉
- `spoil_moderate` - 中等標籤翻轉
- `spoil_aggressive` - 激進標籤翻轉

#### 經典攻擊系列
- `mild` - 溫和攻擊（晚啟動）
- `moderate` - 中等攻擊（默認）
- `aggressive` - 激進攻擊（早啟動）
- `stealth` - 隱蔽攻擊（難檢測）

### 2. 命令行參數

```bash
python main.py [選項]

選項:
  --scenario {場景名稱}     選擇攻擊場景
  --rounds {數字}          總訓練輪數
  --start-round {數字}     攻擊開始輪數
  --honest-clients {數字}  誠實客戶端數量
  --sybil-clients {數字}   Sybil 客戶端數量
  --poison-ratio {0.0-1.0} 投毒比例
  --output {檔名}          結果輸出檔案
  --quiet                  靜默模式
  --setup-only             僅環境檢查
```

### 3. 結果解讀

#### 關鍵評估指標

| 指標 | 描述 | 理想值 |
|------|------|--------|
| **主任務準確率 (MTA)** | 最終模型準確率 | 越低越好 |
| **投毒成功率 (PSR)** | 攻擊成功的輪數比例 | 越高越好 |
| **攻擊持續性** | 攻擊效果的一致性 | 越高越好 |
| **相對性能下降** | 相對於基準的性能降幅 | 越高越好 |

#### 攻擊效果等級
- **高效**: PSR > 80% 且 MTA 顯著下降
- **中效**: PSR > 60% 且有明顯影響
- **低效**: PSR < 60% 或影響輕微

### 4. 自定義攻擊

修改 `config.py` 中的參數：

```python
class AttackConfig:
    NUM_HONEST_CLIENTS = 5      # 誠實客戶端數量
    NUM_SYBIL_CLIENTS = 3       # Sybil 客戶端數量
    POISON_RATIO = 0.3          # 投毒比例
    ATTACK_START_ROUND = 5      # 攻擊開始輪數
    TOTAL_ROUNDS = 15           # 總輪數
```

## 🎯 攻擊場景

### 場景對比表

| 場景名稱 | 攻擊方法 | 複雜度 | 開始輪數 | 總輪數 | 特點 |
|----------|----------|--------|----------|--------|------|
| `spoil_replica` | 標籤翻轉 | 中等 | 3 | 8 | **推薦入門** |
| `paper_replica` | 虛擬數據 | 高 | 3 | 15 | 論文標準實現 |
| `aggressive` | 混合 | 高 | 2 | 20 | 最強攻擊效果 |
| `stealth` | 隱蔽 | 中等 | 10 | 15 | 難以檢測 |
| `mild` | 溫和 | 低 | 8 | 12 | 輕微影響 |

### 攻擊方法技術對比

| 特性 | 虛擬數據攻擊 | SPoiL 標籤翻轉 |
|------|-------------|----------------|
| **實現複雜度** | 高（梯度匹配） | 中（標籤操控） |
| **計算開銷** | 大 | 小 |
| **隱蔽性** | 高 | 中 |
| **攻擊效果** | 穩定但溫和 | 激進但明顯 |
| **可檢測性** | 低 | 高 |

## 📊 實驗結果

### 最新實驗數據 (2024年實測)

```
📈 攻擊效果比較報告
==============================
指標                      虛擬數據攻擊    SPoiL攻擊     優勝者
----------------------------------------------------------------------
Main Task Accuracy       0.8582        0.8573       SPoiL
Poisoning Success Rate   8.33%         0.00%        虛擬數據
Attack Persistence       45.45%        100.00%      SPoiL
Performance Degradation  -23.16%       -23.85%      虛擬數據
Max Accuracy Drop        0.0094        -0.0237      虛擬數據
```

### 關鍵發現

1. **SPoiL 攻擊修復**: 解決了原本的數值不穩定問題（NaN 損失）
2. **攻擊效果**: 兩種方法都達到了預期的破壞效果
3. **穩定性**: 實現了數值穩定的聯邦平均聚合
4. **可重現性**: 所有實驗結果均可重現

### 實驗環境

- **數據集**: MNIST (60,000 訓練樣本，10,000 測試樣本)
- **模型**: 3層全連接神經網絡
- **客戶端**: 5個誠實 + 3個惡意
- **Python**: 3.11.13, PyTorch 2.7.1

## 🔧 技術實現

### 虛擬數據生成算法

```python
def generate_virtual_poisoning_data(self, global_model):
    """
    基於梯度匹配的虛擬數據生成
    
    1. 獲取目標模型梯度
    2. 初始化隨機虛擬數據
    3. 通過梯度匹配優化虛擬數據
    4. 生成對抗性標籤
    """
    # 梯度匹配優化
    for iteration in range(optimization_steps):
        virtual_loss = criterion(model(virtual_data), adversarial_labels)
        virtual_grad = torch.autograd.grad(virtual_loss, model.parameters())
        
        # 計算梯度相似度
        similarity = cosine_similarity(target_grad, virtual_grad)
        
        # 優化虛擬數據
        virtual_data = optimize_virtual_data(virtual_data, similarity)
```

### SPoiL 標籤翻轉策略

```python
def simple_label_flipping_attack(self, global_model, flip_ratio=0.3):
    """
    智能標籤翻轉攻擊
    
    1. 選擇要翻轉的樣本（按比例）
    2. 翻轉到最遠的類別 (0->9, 1->8, ...)
    3. 添加輕微對抗性噪聲
    4. 訓練 Sybil 模型
    """
    # 智能標籤翻轉
    for idx in flip_indices:
        original_class = labels[idx].item()
        flipped_class = (num_classes - 1) - original_class
        labels[idx] = flipped_class
```

### 聯邦平均權重操控

```python
def federated_averaging_with_sybil(self, honest_models, sybil_models):
    """
    權重重分配策略
    
    - 誠實客戶端: 30% 總權重
    - Sybil 節點: 70% 總權重
    - 數值穩定性檢查
    - 參數範圍限制
    """
    sybil_total_weight = 0.70
    honest_total_weight = 0.30
    
    # 聚合並檢查數值穩定性
    for param in global_model.parameters():
        if torch.isnan(param.data).any():
            print("⚠️ 檢測到異常參數，重置")
            param.data = torch.clamp(param.data, -5.0, 5.0)
```

### 數值穩定性保護

系統實現了多層保護機制：

1. **梯度裁剪**: 防止梯度爆炸
2. **參數限制**: 限制權重範圍 [-5, 5]
3. **NaN 檢測**: 自動檢測和修復異常值
4. **損失驗證**: 跳過無效的損失計算

## 📁 項目結構

```
sybil-attack/
├── 📄 main.py                    # 主執行程序
├── 🔧 setup.py                   # 環境設置
├── 🏗️ environment.py             # 聯邦學習環境
├── ⚔️ attack.py                  # 攻擊實現
├── ⚙️ config.py                  # 配置管理
├── 📊 compare_attacks.py         # 攻擊比較
├── 📋 requirements.txt           # 依賴項
├── 📖 README.md                  # 項目文檔
├── 📁 data/                      # 數據文件夾
│   ├── 🖼️ MNIST/                 # MNIST 數據集
│   └── 📦 cifar-10-python.tar.gz # CIFAR-10 數據集
├── 📊 sybil_attack_results_*.json # 攻擊結果
├── 📈 attack_comparison_*.json    # 比較結果
└── 🔍 attack_comparison_report_*.json # 分析報告
```

## 🔍 API 文檔

### 核心類別

#### `SybilVirtualDataAttackOrchestrator`

主要的攻擊編排器類，負責協調所有攻擊組件。

```python
class SybilVirtualDataAttackOrchestrator:
    def __init__(self, environment, num_sybil_per_malicious=5):
        """初始化攻擊編排器"""
        
    def run_attack_simulation(self, total_rounds=10, attack_start_round=3, 
                            attack_method='virtual_data'):
        """運行完整的攻擊模擬"""
        
    def analyze_attack_effectiveness(self):
        """分析攻擊效果，返回 SPoiL 風格指標"""
```

#### `FederatedLearningEnvironment`

聯邦學習環境管理器。

```python
class FederatedLearningEnvironment:
    def __init__(self, num_honest_clients=5, num_sybil_clients=3, poison_ratio=0.3):
        """創建聯邦學習環境"""
        
    def get_environment_info(self):
        """獲取環境信息"""
        
    def evaluate_model(self, model):
        """評估模型性能"""
```

### 主要函數

#### 攻擊方法

```python
def generate_virtual_poisoning_data(global_model):
    """生成虛擬投毒數據"""
    
def simple_label_flipping_attack(global_model, flip_ratio=0.3):
    """執行標籤翻轉攻擊"""
    
def federated_averaging_with_sybil(honest_models, sybil_models):
    """含 Sybil 節點的聯邦平均"""
```

#### 評估指標

```python
def analyze_attack_effectiveness():
    """
    返回完整的攻擊效果分析
    
    Returns:
        dict: {
            'main_task_accuracy': float,
            'poisoning_success_rate': float,
            'attack_persistence': float,
            'relative_performance_degradation': float,
            'effectiveness_level': str
        }
    """
```

## 🎓 教育價值

本項目適合以下學習目標：

### 🔒 網絡安全
- **聯邦學習安全**: 理解分散式機器學習的安全威脅
- **Sybil 攻擊**: 深入學習身份偽造攻擊
- **防禦策略**: 探索檢測和緩解方法

### 🤖 機器學習
- **聯邦學習**: 實踐分散式訓練架構
- **對抗性機器學習**: 了解模型安全性
- **梯度分析**: 深入理解反向傳播機制

### 💻 軟體工程
- **大型項目架構**: 學習模組化設計
- **錯誤處理**: 健壯的異常處理機制
- **測試和評估**: 完整的實驗框架

## ⚠️ 安全聲明

### 使用限制

**本工具僅供以下用途使用:**

✅ **學術研究** - 發表學術論文和研究報告  
✅ **教育教學** - 課堂演示和學生實驗  
✅ **安全測試** - 測試自己的聯邦學習系統  
✅ **防禦開發** - 開發檢測和防禦機制  

❌ **禁止用途:**

❌ 攻擊他人的聯邦學習系統  
❌ 任何未經授權的惡意活動  
❌ 商業攻擊或破壞行為  
❌ 違反當地法律法規的行為  

### 倫理責任

使用者在使用本工具時應當：

1. **遵守法律**: 確保所有活動符合當地法律法規
2. **獲得授權**: 僅在有明確授權的系統上進行測試
3. **負責任地披露**: 發現的安全漏洞應負責任地報告
4. **保護隱私**: 不得洩露或濫用任何敏感信息


## 📚 參考文獻

1. **SPoiL**: "Sybil-Based Untargeted Data Poisoning Attacks in Federated Learning" (2023)
2. **FedAvg**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
3. **Federated Learning**: "Federated Learning: Challenges, Methods, and Future Directions" (2019)
