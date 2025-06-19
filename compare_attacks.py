#!/usr/bin/env python3
"""
攻擊方法比較腳本
================

比較我們的虛擬數據攻擊與 SPoiL 風格的標籤翻轉攻擊
使用 Main Task Accuracy 和 Poisoning Success Rate 評估

Author: Security Research Team
Date: 2024
"""

import json
import sys
from datetime import datetime
from pathlib import Path

def main():
    """主函數 - 運行攻擊比較實驗"""
    
    print("🔬 攻擊方法比較實驗")
    print("=" * 70)
    print("📊 比較目標:")
    print("   1. 虛擬數據攻擊 (我們的實現)")
    print("   2. SPoiL 標籤翻轉攻擊 (2023論文)")
    print("📈 評估指標:")
    print("   - Main Task Accuracy (MTA)")
    print("   - Poisoning Success Rate (PSR)")
    print("   - Attack Persistence")
    print("   - Relative Performance Degradation")
    print("=" * 70)
    
    try:
        from setup import quick_setup
        from environment import FederatedLearningEnvironment
        from attack import create_attack_orchestrator, ATTACK_SCENARIOS
        
        # 環境設置
        print("🔧 初始化環境...")
        setup_success, config = quick_setup()
        if not setup_success:
            print("❌ 環境設置失敗")
            sys.exit(1)
        
        # 創建聯邦學習環境
        fl_env = FederatedLearningEnvironment(
            num_honest_clients=5,
            num_sybil_clients=3,
            poison_ratio=0.3
        )
        
        # 攻擊場景配置
        attack_scenarios = [
            ('paper_replica', '虛擬數據攻擊 (論文實現)'),
            ('spoil_replica', 'SPoiL 標籤翻轉攻擊 (2023)')
        ]
        
        results_comparison = []
        
        for scenario_name, description in attack_scenarios:
            print(f"\n🎯 執行攻擊: {description}")
            print("-" * 50)
            
            # 重新創建環境以確保公平比較
            fl_env = FederatedLearningEnvironment(
                num_honest_clients=5,
                num_sybil_clients=3,
                poison_ratio=0.3
            )
            
            # 創建攻擊編排器
            scenario_config = ATTACK_SCENARIOS[scenario_name]
            attack_orchestrator = create_attack_orchestrator(
                fl_env, 
                num_sybil_per_malicious=scenario_config.get('num_sybil_per_malicious', 5)
            )
            
            # 執行攻擊
            results = attack_orchestrator.run_attack_simulation(
                total_rounds=scenario_config['total_rounds'],
                attack_start_round=scenario_config['attack_start_round'],
                attack_method=scenario_config.get('attack_method', 'virtual_data'),
                verbose=True
            )
            
            # 保存結果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attack_comparison_{scenario_name}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 結果已保存: {filename}")
            
            # 收集比較數據
            attack_results = results['results']
            results_comparison.append({
                'scenario': scenario_name,
                'description': description,
                'attack_method': scenario_config.get('attack_method', 'virtual_data'),
                'metrics': {
                    'main_task_accuracy': attack_results.get('main_task_accuracy', 0),
                    'poisoning_success_rate': attack_results.get('poisoning_success_rate', 0),
                    'attack_persistence': attack_results.get('attack_persistence', 0),
                    'relative_performance_degradation': attack_results.get('relative_performance_degradation', 0),
                    'max_accuracy_drop': attack_results.get('max_accuracy_drop', 0),
                    'effectiveness_level': attack_results.get('effectiveness_level', 'unknown')
                },
                'config': {
                    'total_rounds': scenario_config['total_rounds'],
                    'attack_start_round': scenario_config['attack_start_round'],
                    'num_sybil_per_malicious': scenario_config.get('num_sybil_per_malicious', 5)
                }
            })
        
        # 生成比較報告
        print("\n" + "=" * 70)
        print("📊 攻擊效果比較報告")
        print("=" * 70)
        
        print("\n📈 核心指標對比:")
        print("-" * 70)
        print(f"{'指標':<25} {'虛擬數據攻擊':<15} {'SPoiL攻擊':<15} {'優勢'}")
        print("-" * 70)
        
        if len(results_comparison) >= 2:
            virtual_data = results_comparison[0]['metrics']
            spoil_attack = results_comparison[1]['metrics']
            
            # Main Task Accuracy (越低越好，表示攻擊越有效)
            print(f"{'Main Task Accuracy':<25} {virtual_data['main_task_accuracy']:<15.4f} {spoil_attack['main_task_accuracy']:<15.4f} {'SPoiL' if spoil_attack['main_task_accuracy'] < virtual_data['main_task_accuracy'] else '虛擬數據'}")
            
            # Poisoning Success Rate (越高越好)
            print(f"{'Poisoning Success Rate':<25} {virtual_data['poisoning_success_rate']:<15.2%} {spoil_attack['poisoning_success_rate']:<15.2%} {'SPoiL' if spoil_attack['poisoning_success_rate'] > virtual_data['poisoning_success_rate'] else '虛擬數據'}")
            
            # Attack Persistence (越高越好)
            print(f"{'Attack Persistence':<25} {virtual_data['attack_persistence']:<15.2%} {spoil_attack['attack_persistence']:<15.2%} {'SPoiL' if spoil_attack['attack_persistence'] > virtual_data['attack_persistence'] else '虛擬數據'}")
            
            # Relative Performance Degradation (越高越好)
            print(f"{'Performance Degradation':<25} {virtual_data['relative_performance_degradation']:<15.2%} {spoil_attack['relative_performance_degradation']:<15.2%} {'SPoiL' if spoil_attack['relative_performance_degradation'] > virtual_data['relative_performance_degradation'] else '虛擬數據'}")
            
            # Max Accuracy Drop (越高越好)
            print(f"{'Max Accuracy Drop':<25} {virtual_data['max_accuracy_drop']:<15.4f} {spoil_attack['max_accuracy_drop']:<15.4f} {'SPoiL' if spoil_attack['max_accuracy_drop'] > virtual_data['max_accuracy_drop'] else '虛擬數據'}")
        
        print("\n🎯 效果等級對比:")
        for result in results_comparison:
            print(f"   {result['description']}: {result['metrics']['effectiveness_level']}")
        
        print("\n📋 攻擊配置對比:")
        for result in results_comparison:
            config = result['config']
            print(f"\n{result['description']}:")
            print(f"   - 總輪數: {config['total_rounds']}")
            print(f"   - 攻擊開始輪: {config['attack_start_round']}")
            print(f"   - Sybil節點數: {config['num_sybil_per_malicious']}")
            print(f"   - 攻擊方法: {result['attack_method']}")
        
        # 保存比較結果
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'comparison_results': results_comparison,
            'summary': {
                'virtual_data_effectiveness': results_comparison[0]['metrics']['effectiveness_level'] if len(results_comparison) > 0 else 'unknown',
                'spoil_effectiveness': results_comparison[1]['metrics']['effectiveness_level'] if len(results_comparison) > 1 else 'unknown',
            }
        }
        
        comparison_filename = f"attack_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_filename, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 完整比較報告已保存: {comparison_filename}")
        
        print("\n🔍 結論:")
        if len(results_comparison) >= 2:
            virtual_psr = results_comparison[0]['metrics']['poisoning_success_rate']
            spoil_psr = results_comparison[1]['metrics']['poisoning_success_rate']
            
            if virtual_psr > spoil_psr:
                print("   虛擬數據攻擊在投毒成功率方面表現更優")
            elif spoil_psr > virtual_psr:
                print("   SPoiL 標籤翻轉攻擊在投毒成功率方面表現更優")
            else:
                print("   兩種攻擊方法在投毒成功率方面表現相當")
        
        print("\n🎉 攻擊比較實驗完成!")
        
    except Exception as e:
        print(f"❌ 實驗過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 