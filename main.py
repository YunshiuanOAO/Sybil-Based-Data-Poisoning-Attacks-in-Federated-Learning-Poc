#!/usr/bin/env python3
"""
Sybil 攻擊主執行腳本
==================

這是執行 Sybil 攻擊模擬的主入口點。

使用方法:
  python main.py                    # 運行默認攻擊場景
  python main.py --scenario mild    # 運行溫和攻擊場景
  python main.py --help            # 顯示幫助信息

Author: Security Research Team
Date: 2024
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='Sybil 攻擊模擬工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
攻擊場景選項:
  mild       - 溫和攻擊 (較晚開始，影響較小)
  moderate   - 中等攻擊 (中期開始，平衡影響) [默認]
  aggressive - 激進攻擊 (早期開始，持續時間長)  
  stealth    - 隱蔽攻擊 (很晚才開始，難以被發現)

範例:
  python main.py
  python main.py --scenario aggressive
  python main.py --rounds 15 --start-round 2
  python main.py --honest-clients 8 --sybil-clients 4
        """
    )
    
    parser.add_argument(
        '--scenario', 
        choices=['mild', 'moderate', 'aggressive', 'stealth', 'mild_virtual', 'moderate_virtual', 'aggressive_virtual', 'stealth_virtual', 'paper_replica', 'spoil_mild', 'spoil_moderate', 'spoil_aggressive', 'spoil_replica', 'spoil_original'], 
        default='moderate',
        help='選擇攻擊場景 (默認: moderate, 虛擬數據攻擊: paper_replica, SPoiL攻擊: spoil_replica)'
    )
    
    parser.add_argument(
        '--rounds', 
        type=int, 
        default=None,
        help='訓練輪數 (覆蓋場景默認值)'
    )
    
    parser.add_argument(
        '--start-round', 
        type=int, 
        default=None,
        help='攻擊開始輪數 (覆蓋場景默認值)'
    )
    
    parser.add_argument(
        '--honest-clients', 
        type=int, 
        default=5,
        help='誠實客戶端數量 (默認: 5)'
    )
    
    parser.add_argument(
        '--sybil-clients', 
        type=int, 
        default=3,
        help='Sybil 客戶端數量 (默認: 3)'
    )
    
    parser.add_argument(
        '--poison-ratio', 
        type=float, 
        default=0.3,
        help='投毒比例 (0.0-1.0, 默認: 0.3)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='結果輸出文件名 (默認: sybil_attack_results_TIMESTAMP.json)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='靜默模式，減少輸出'
    )
    
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='僅進行環境設置檢查'
    )
    
    args = parser.parse_args()
    
    # 顯示歡迎信息
    if not args.quiet:
        print_banner()
    
    # 1. 環境設置階段
    print("🔧 階段 1: 環境設置")
    print("-" * 50)
    
    try:
        from setup import quick_setup, validate_environment
        
        # 執行環境設置
        setup_success, config = quick_setup()
        
        if not setup_success:
            print("❌ 環境設置失敗，無法繼續")
            sys.exit(1)
            
        # 如果只是設置檢查，這裡就結束
        if args.setup_only:
            print("\n✅ 環境設置檢查完成!")
            return
            
    except Exception as e:
        print(f"❌ 環境設置過程中出現錯誤: {e}")
        print("請檢查是否正確安裝了所有依賴項")
        sys.exit(1)
    
    # 2. 創建環境階段
    print("\n🏗️ 階段 2: 創建聯邦學習環境")
    print("-" * 50)
    
    try:
        from environment import FederatedLearningEnvironment
        
        # 創建環境
        fl_env = FederatedLearningEnvironment(
            num_honest_clients=args.honest_clients,
            num_sybil_clients=args.sybil_clients,
            poison_ratio=args.poison_ratio
        )
        
        env_info = fl_env.get_environment_info()
        if not args.quiet:
            print(f"✅ 環境創建成功:")
            print(f"   誠實客戶端: {env_info['num_honest_clients']}")
            print(f"   Sybil客戶端: {env_info['num_sybil_clients']}")
            print(f"   Sybil比例: {env_info['sybil_ratio']:.2%}")
            
    except Exception as e:
        print(f"❌ 環境創建失敗: {e}")
        sys.exit(1)
    
    # 3. 攻擊執行階段
    print("\n🎯 階段 3: 執行 Sybil 攻擊")
    print("-" * 50)
    
    try:
        from attack import SybilVirtualDataAttackOrchestrator, ATTACK_SCENARIOS, create_attack_orchestrator
        
        # 創建虛擬數據攻擊編排器
        attack_orchestrator = create_attack_orchestrator(
            fl_env, 
            num_sybil_per_malicious=ATTACK_SCENARIOS[args.scenario].get('num_sybil_per_malicious', 5)
        )
        
        # 🆕 設置當前場景
        attack_orchestrator.set_current_scenario(args.scenario)
        
        # 獲取攻擊參數
        scenario_config = ATTACK_SCENARIOS[args.scenario]
        total_rounds = args.rounds if args.rounds else scenario_config['total_rounds']
        start_round = args.start_round if args.start_round else scenario_config['attack_start_round']
        attack_method = scenario_config.get('attack_method', 'virtual_data')
        
        if not args.quiet:
            print(f"📋 攻擊配置:")
            print(f"   場景: {args.scenario} ({scenario_config['description']})")
            print(f"   總輪數: {total_rounds}")
            print(f"   攻擊開始輪數: {start_round}")
            print(f"   攻擊方法: {attack_method}")
        
        # 執行攻擊模擬
        results = attack_orchestrator.run_attack_simulation(
            total_rounds=total_rounds,
            attack_start_round=start_round,
            attack_method=attack_method,
            verbose=not args.quiet
        )
        
    except Exception as e:
        print(f"❌ 攻擊執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. 結果保存階段
    print("\n💾 階段 4: 保存結果")
    print("-" * 50)
    
    try:
        # 生成輸出文件名
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sybil_attack_results_{timestamp}.json"
        
        # 保存結果
        saved_file = attack_orchestrator.save_results(output_file)
        
        # 顯示可視化
        if not args.quiet:
            attack_orchestrator.visualize_attack_progress()
            
    except Exception as e:
        print(f"⚠️ 結果保存失敗: {e}")
        print("但攻擊模擬已完成")
    
    # 5. 總結
    if not args.quiet:
        print("\n📊 執行總結")
        print("-" * 50)
        print(f"✅ 攻擊模擬完成!")
        print(f"   最終準確率: {results.get('final_accuracy', 0):.4f}")
        print(f"   攻擊效果: {results['results'].get('effectiveness_level', 'Unknown')}")
        print(f"   結果文件: {output_file}")
        print("\n🎉 感謝使用 Sybil 攻擊模擬工具!")

def print_banner():
    """打印歡迎橫幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Sybil 攻擊模擬工具                         ║
║                                                              ║
║  🎯 專業的聯邦學習安全研究工具                                ║
║  🔬 教育和研究目的使用                                       ║
║  ⚠️  請勿用於惡意攻擊                                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用戶中斷，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序執行出現未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 