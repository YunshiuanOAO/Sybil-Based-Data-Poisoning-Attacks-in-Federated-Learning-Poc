#!/usr/bin/env python3
"""
Simple script to run sybil-based untargeted poisoning attacks
============================================================

This script provides a command-line interface for running the attack tool
with different configurations and scenarios.

Usage:
    python run_attack.py --scenario moderate --rounds 15 --start-round 5
"""

import argparse
import torch
import numpy as np
import random
from sybil_attack_tool import FederatedLearningEnvironment, SybilAttackTool
from config import AttackConfig, AttackScenarios

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sybil-Based Untargeted Poisoning Attack Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios:
  mild      - Low impact attack (2 sybil clients, 20% poison ratio)
  moderate  - Balanced attack (3 sybil clients, 30% poison ratio)
  severe    - High impact attack (5 sybil clients, 50% poison ratio)
  stealth   - Hard-to-detect attack (2 sybil clients, 15% poison ratio)
  custom    - Use configuration from config.py

Examples:
  python run_attack.py --scenario moderate
  python run_attack.py --scenario severe --rounds 20 --start-round 8
  python run_attack.py --honest-clients 8 --sybil-clients 4 --rounds 15
        """
    )
    
    parser.add_argument(
        '--scenario', 
        choices=['mild', 'moderate', 'severe', 'stealth', 'custom'],
        default='moderate',
        help='Predefined attack scenario (default: moderate)'
    )
    
    parser.add_argument(
        '--honest-clients',
        type=int,
        help='Number of honest clients (overrides scenario)'
    )
    
    parser.add_argument(
        '--sybil-clients',
        type=int,
        help='Number of sybil clients (overrides scenario)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=15,
        help='Total number of federated learning rounds (default: 15)'
    )
    
    parser.add_argument(
        '--start-round',
        type=int,
        default=5,
        help='Round when attack begins (default: 5)'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['MNIST'],
        default='MNIST',
        help='Dataset to use (default: MNIST)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()

def configure_scenario(scenario):
    """Configure attack parameters based on scenario"""
    scenario_map = {
        'mild': AttackScenarios.mild_attack,
        'moderate': AttackScenarios.moderate_attack,
        'severe': AttackScenarios.severe_attack,
        'stealth': AttackScenarios.stealth_attack,
        'custom': lambda: None  # Use existing config
    }
    
    if scenario in scenario_map:
        scenario_map[scenario]()
        return True
    return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Print header
    print("🎯 Sybil-Based Untargeted Poisoning Attack Tool")
    print("🚨 Educational and Research Purposes Only")
    print("=" * 60)
    
    # Set random seed
    set_seeds(args.seed)
    print(f"🌱 Random seed set to: {args.seed}")
    
    # Configure scenario
    if not configure_scenario(args.scenario):
        print(f"❌ Unknown scenario: {args.scenario}")
        return 1
    
    # Override with command line arguments if provided
    if args.honest_clients:
        AttackConfig.NUM_HONEST_CLIENTS = args.honest_clients
    if args.sybil_clients:
        AttackConfig.NUM_SYBIL_CLIENTS = args.sybil_clients
    
    AttackConfig.DATASET_NAME = args.dataset
    
    # Print configuration
    print(f"\n📋 Attack Configuration:")
    print(f"  Scenario: {args.scenario}")
    print(f"  Honest Clients: {AttackConfig.NUM_HONEST_CLIENTS}")
    print(f"  Sybil Clients: {AttackConfig.NUM_SYBIL_CLIENTS}")
    print(f"  Sybil Ratio: {AttackConfig.NUM_SYBIL_CLIENTS/(AttackConfig.NUM_HONEST_CLIENTS + AttackConfig.NUM_SYBIL_CLIENTS):.1%}")
    print(f"  Dataset: {AttackConfig.DATASET_NAME}")
    print(f"  Total Rounds: {args.rounds}")
    print(f"  Attack Start Round: {args.start_round}")
    print(f"  Poison Ratio: {AttackConfig.POISON_RATIO:.1%}")
    
    # Create federated learning environment
    print(f"\n🏗️  Creating federated learning environment...")
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=AttackConfig.NUM_HONEST_CLIENTS,
        num_sybil_clients=AttackConfig.NUM_SYBIL_CLIENTS,
        dataset_name=AttackConfig.DATASET_NAME
    )
    
    # Create attack tool
    attack_tool = SybilAttackTool(fl_env)
    
    # Conduct attack
    print(f"\n🚀 Starting attack simulation...")
    try:
        attack_tool.conduct_attack(
            num_rounds=args.rounds,
            attack_start_round=args.start_round
        )
        
        # Print final results
        history = fl_env.training_history
        final_accuracy = history['accuracy'][-1]
        max_degradation = max([h - a for h, a in zip(history['honest_accuracy'], history['accuracy'])])
        
        print(f"\n📈 Final Results:")
        print(f"  Final Model Accuracy: {final_accuracy:.4f}")
        print(f"  Maximum Accuracy Drop: {max_degradation:.4f}")
        print(f"  Attack Effectiveness: {'High' if max_degradation > 0.15 else 'Moderate' if max_degradation > 0.05 else 'Low'}")
        
        if not args.no_plots:
            print(f"\n📊 Analysis saved to:")
            print(f"  • sybil_attack_analysis.png (visualization)")
            print(f"  • sybil_attack_results.csv (detailed data)")
        
        print(f"\n✅ Attack simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during attack simulation: {str(e)}")
        return 1
    
    print(f"\n⚠️  Remember: This tool is for educational and research purposes only.")
    return 0

if __name__ == "__main__":
    exit(main()) 