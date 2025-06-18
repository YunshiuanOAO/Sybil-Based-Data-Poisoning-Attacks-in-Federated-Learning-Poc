"""
環境設置模組
============

此模組負責：
- 檢查系統依賴
- 初始化實驗環境
- 驗證配置參數
- 準備數據集

Author: Security Research Team
Date: 2024
"""

import os
import sys
import importlib
import warnings
from typing import Dict, List, Tuple, Any
from pathlib import Path

# 抑制警告信息
warnings.filterwarnings('ignore')

class EnvironmentSetup:
    """環境設置類"""
    
    def __init__(self):
        self.required_modules = [
            'torch', 'torchvision', 'numpy', 'json', 'datetime'
        ]
        self.optional_modules = [
            'matplotlib', 'seaborn', 'pandas'
        ]
        self.setup_status = {
            'dependencies': False,
            'data_directory': False,
            'config_loaded': False
        }
        
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """檢查必要的依賴項"""
        missing_modules = []
        
        print("🔍 檢查系統依賴...")
        
        for module in self.required_modules:
            try:
                if module == 'torch':
                    import torch
                    print(f"  ✅ PyTorch {torch.__version__}")
                elif module == 'torchvision':
                    import torchvision
                    print(f"  ✅ Torchvision {torchvision.__version__}")
                elif module == 'numpy':
                    import numpy as np
                    print(f"  ✅ NumPy {np.__version__}")
                else:
                    importlib.import_module(module)
                    print(f"  ✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"  ❌ {module} (缺失)")
        
        # 檢查可選模組
        print("\n📊 檢查可選依賴...")
        for module in self.optional_modules:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}")
            except ImportError:
                print(f"  ⚠️ {module} (可選，用於高級可視化)")
        
        all_required_available = len(missing_modules) == 0
        self.setup_status['dependencies'] = all_required_available
        
        return all_required_available, missing_modules
    
    def setup_data_directory(self) -> bool:
        """設置數據目錄"""
        print("\n📁 設置數據目錄...")
        
        data_dir = Path('./data')
        try:
            data_dir.mkdir(exist_ok=True)
            print(f"  ✅ 數據目錄: {data_dir.absolute()}")
            self.setup_status['data_directory'] = True
            return True
        except Exception as e:
            print(f"  ❌ 無法創建數據目錄: {e}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """加載配置"""
        print("\n⚙️ 加載配置...")
        
        try:
            from config import AttackConfig
            config = AttackConfig()
            print("  ✅ 配置加載成功")
            self.setup_status['config_loaded'] = True
            return config
        except ImportError:
            print("  ⚠️ 未找到config.py，使用默認配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> object:
        """獲取默認配置"""
        class DefaultConfig:
            NUM_HONEST_CLIENTS = 5
            NUM_SYBIL_CLIENTS = 3
            DATASET_NAME = 'MNIST'
            LEARNING_RATE_HONEST = 0.01
            LEARNING_RATE_SYBIL = 0.05
            POISON_RATIO = 0.3
            TOTAL_ROUNDS = 12
            ATTACK_START_ROUND = 3
            
        return DefaultConfig()
    
    def test_torch_functionality(self) -> bool:
        """測試 PyTorch 基本功能"""
        print("\n🧪 測試 PyTorch 功能...")
        
        try:
            import torch
            import torch.nn as nn
            
            # 測試基本張量操作
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            z = x + y
            print(f"  ✅ 張量操作: {z.shape}")
            
            # 測試神經網絡模組
            model = nn.Linear(3, 1)
            output = model(x)
            print(f"  ✅ 神經網絡: {output.shape}")
            
            # 測試 CUDA 可用性
            if torch.cuda.is_available():
                print(f"  ✅ CUDA 可用: {torch.cuda.get_device_name()}")
            else:
                print("  ℹ️ CUDA 不可用，將使用 CPU")
                
            return True
            
        except Exception as e:
            print(f"  ❌ PyTorch 測試失敗: {e}")
            return False
    
    def display_system_info(self):
        """顯示系統信息"""
        print("\n💻 系統信息:")
        print(f"  Python 版本: {sys.version}")
        print(f"  操作系統: {os.name}")
        print(f"  工作目錄: {os.getcwd()}")
        
        try:
            import torch
            print(f"  PyTorch 版本: {torch.__version__}")
            print(f"  CUDA 可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA 版本: {torch.version.cuda}")
                print(f"  GPU 數量: {torch.cuda.device_count()}")
        except ImportError:
            print("  PyTorch: 未安裝")
    
    def run_complete_setup(self) -> Tuple[bool, object]:
        """運行完整的環境設置"""
        print("🚀 開始環境設置")
        print("=" * 50)
        
        # 顯示系統信息
        self.display_system_info()
        
        # 檢查依賴項
        deps_ok, missing = self.check_dependencies()
        if not deps_ok:
            print(f"\n❌ 缺失必要依賴: {missing}")
            print("請安裝缺失的模組後重試")
            return False, None
            
        # 設置數據目錄
        if not self.setup_data_directory():
            print("❌ 數據目錄設置失敗")
            return False, None
            
        # 測試 PyTorch 功能
        if not self.test_torch_functionality():
            print("❌ PyTorch 功能測試失敗")
            return False, None
            
        # 加載配置
        config = self.load_config()
        
        print("\n✅ 環境設置完成!")
        print("=" * 50)
        
        return True, config
    
    def get_setup_status(self) -> Dict[str, bool]:
        """獲取設置狀態"""
        return self.setup_status.copy()

def quick_setup() -> Tuple[bool, object]:
    """快速環境設置"""
    setup = EnvironmentSetup()
    return setup.run_complete_setup()

def validate_environment() -> bool:
    """驗證環境是否正確設置"""
    setup = EnvironmentSetup()
    
    # 檢查必要文件
    required_files = ['environment.py', 'attack.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ 缺失必要文件: {file}")
            return False
            
    # 檢查依賴項
    deps_ok, _ = setup.check_dependencies()
    if not deps_ok:
        return False
        
    # 測試導入
    try:
        from environment import FederatedLearningEnvironment
        from attack import SybilAttackOrchestrator
        print("✅ 模組導入測試通過")
        return True
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        return False

def install_requirements():
    """安裝依賴項指南"""
    print("📦 依賴項安裝指南:")
    print("=" * 40)
    print("1. 使用 pip 安裝:")
    print("   pip install torch torchvision numpy")
    print("\n2. 或使用 requirements.txt:")
    print("   pip install -r requirements.txt")
    print("\n3. 對於 CUDA 支持:")
    print("   請訪問 https://pytorch.org/ 獲取適合您系統的安裝命令")
    print("=" * 40)

if __name__ == "__main__":
    # 如果直接運行此腳本，執行環境設置
    success, config = quick_setup()
    
    if success:
        print("\n🎉 環境設置成功! 您現在可以運行攻擊腳本了。")
        print("\n使用方法:")
        print("  python main.py")
    else:
        print("\n❌ 環境設置失敗，請檢查上述錯誤信息。")
        install_requirements() 