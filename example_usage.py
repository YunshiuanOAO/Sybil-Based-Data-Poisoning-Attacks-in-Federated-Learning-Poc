"""
Example Usage of Sybil-Based Untargeted Poisoning Attack Tool
=============================================================

This script demonstrates different ways to use the sybil attack tool
with various configurations and scenarios.
"""

from sybil_attack_tool import *
from config import AttackConfig, AttackScenarios
import torch
import numpy as np
import random

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def example_1_basic_attack():
    """Example 1: Basic sybil attack with default settings"""
    print("🔥 Example 1: Basic Sybil Attack")
    print("=" * 50)
    
    set_seeds(42)
    
    # Create federated learning environment
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=5,
        num_sybil_clients=3,
        dataset_name='MNIST'
    )
    
    # Create and run attack
    attack_tool = SybilAttackTool(fl_env)
    attack_tool.conduct_attack(num_rounds=10, attack_start_round=3)

def example_2_configurable_attack():
    """Example 2: Using configuration file for customized attack"""
    print("\n🔥 Example 2: Configurable Sybil Attack")
    print("=" * 50)
    
    set_seeds(123)
    
    # Use moderate attack scenario
    AttackScenarios.moderate_attack()
    print(AttackConfig.get_attack_description())
    
    # Create environment with config
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=AttackConfig.NUM_HONEST_CLIENTS,
        num_sybil_clients=AttackConfig.NUM_SYBIL_CLIENTS,
        dataset_name=AttackConfig.DATASET_NAME
    )
    
    # Run attack with config parameters
    attack_tool = SybilAttackTool(fl_env)
    attack_tool.conduct_attack(
        num_rounds=AttackConfig.TOTAL_ROUNDS,
        attack_start_round=AttackConfig.ATTACK_START_ROUND
    )

def example_3_comparison_study():
    """Example 3: Compare different attack intensities"""
    print("\n🔥 Example 3: Attack Intensity Comparison")
    print("=" * 50)
    
    scenarios = [
        ("Mild Attack", AttackScenarios.mild_attack),
        ("Moderate Attack", AttackScenarios.moderate_attack),
        ("Severe Attack", AttackScenarios.severe_attack),
        ("Stealth Attack", AttackScenarios.stealth_attack)
    ]
    
    results = {}
    
    for scenario_name, scenario_func in scenarios:
        print(f"\n--- Running {scenario_name} ---")
        
        # Configure scenario
        scenario_func()
        set_seeds(456)
        
        # Create environment
        fl_env = FederatedLearningEnvironment(
            num_honest_clients=AttackConfig.NUM_HONEST_CLIENTS,
            num_sybil_clients=AttackConfig.NUM_SYBIL_CLIENTS,
            dataset_name=AttackConfig.DATASET_NAME
        )
        
        # Run attack
        attack_tool = SybilAttackTool(fl_env)
        attack_tool.conduct_attack(
            num_rounds=10,  # Shorter for comparison
            attack_start_round=AttackConfig.ATTACK_START_ROUND
        )
        
        # Store results
        history = fl_env.training_history
        final_accuracy = history['accuracy'][-1]
        max_degradation = max([h - a for h, a in zip(history['honest_accuracy'], history['accuracy'])])
        
        results[scenario_name] = {
            'final_accuracy': final_accuracy,
            'max_degradation': max_degradation,
            'sybil_ratio': AttackConfig.NUM_SYBIL_CLIENTS / (AttackConfig.NUM_HONEST_CLIENTS + AttackConfig.NUM_SYBIL_CLIENTS)
        }
    
    # Print comparison
    print("\n📊 Attack Comparison Results:")
    print("=" * 60)
    print(f"{'Scenario':<20} {'Final Acc':<12} {'Max Degrad':<12} {'Sybil %':<10}")
    print("-" * 60)
    for scenario, data in results.items():
        print(f"{scenario:<20} {data['final_accuracy']:<12.4f} {data['max_degradation']:<12.4f} {data['sybil_ratio']:<10.1%}")

def example_4_custom_poisoning():
    """Example 4: Custom poisoning strategy"""
    print("\n🔥 Example 4: Custom Poisoning Strategy")
    print("=" * 50)
    
    class CustomSybilClient(SybilClient):
        """Custom sybil client with advanced poisoning techniques"""
        
        def _create_poisoned_dataset(self, original_dataset):
            """Enhanced poisoning with multiple strategies"""
            sample_size = min(1500, len(original_dataset) // 8)
            indices = random.sample(range(len(original_dataset)), sample_size)
            
            poisoned_samples = []
            poisoned_labels = []
            
            for idx in indices:
                data, label = original_dataset[idx]
                
                if random.random() < self.poison_ratio:
                    # Strategy 1: Label flipping to adjacent class
                    poisoned_label = (label + 1) % self.num_classes
                    
                    # Strategy 2: Gaussian noise with varying intensity
                    noise_intensity = random.uniform(0.05, 0.15)
                    noise = torch.randn_like(data) * noise_intensity
                    poisoned_data = torch.clamp(data + noise, 0, 1)
                    
                    # Strategy 3: Pixel corruption for certain samples
                    if random.random() < 0.2:
                        corruption_mask = torch.rand_like(data) < 0.1
                        poisoned_data[corruption_mask] = torch.rand_like(poisoned_data[corruption_mask])
                    
                    poisoned_samples.append(poisoned_data)
                    poisoned_labels.append(poisoned_label)
                else:
                    poisoned_samples.append(data)
                    poisoned_labels.append(label)
                    
            return PoisonedDataset(poisoned_samples, poisoned_labels)
    
    # Use custom sybil client
    set_seeds(789)
    
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=6,
        num_sybil_clients=2,
        dataset_name='MNIST'
    )
    
    # Replace default sybil clients with custom ones
    fl_env.sybil_clients = []
    for i in range(2):
        custom_client = CustomSybilClient(
            client_id=f"custom_sybil_{i}",
            original_dataset=fl_env.train_dataset,
            input_size=fl_env.input_size,
            num_classes=fl_env.num_classes,
            poison_ratio=0.4
        )
        fl_env.sybil_clients.append(custom_client)
    
    # Run attack
    attack_tool = SybilAttackTool(fl_env)
    attack_tool.conduct_attack(num_rounds=12, attack_start_round=4)

def example_5_defense_evaluation():
    """Example 5: Evaluate attack against simple defense mechanism"""
    print("\n🔥 Example 5: Attack vs Defense Evaluation")
    print("=" * 50)
    
    class DefensiveAggregation(SybilAttackTool):
        """Attack tool with defensive aggregation mechanism"""
        
        def _federated_averaging(self, local_models):
            """Defensive aggregation using trimmed mean"""
            if len(local_models) <= 2:
                return super()._federated_averaging(local_models)
            
            # Calculate parameter distances from median
            all_params = []
            for model in local_models:
                params = torch.cat([p.flatten() for p in model.parameters()])
                all_params.append(params)
            
            param_stack = torch.stack(all_params)
            param_median = torch.median(param_stack, dim=0)[0]
            
            # Calculate distances
            distances = [torch.norm(params - param_median) for params in all_params]
            
            # Remove outliers (potential sybil clients)
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
            trimmed_indices = sorted_indices[:len(local_models) // 2 + 1]  # Keep majority
            
            trimmed_models = [local_models[i] for i in trimmed_indices]
            
            print(f"  🛡️  Defense: Filtered {len(local_models) - len(trimmed_models)} suspicious models")
            
            return super()._federated_averaging(trimmed_models)
    
    set_seeds(999)
    
    # Configure strong attack
    AttackScenarios.severe_attack()
    
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=AttackConfig.NUM_HONEST_CLIENTS,
        num_sybil_clients=AttackConfig.NUM_SYBIL_CLIENTS,
        dataset_name=AttackConfig.DATASET_NAME
    )
    
    # Use defensive aggregation
    defensive_tool = DefensiveAggregation(fl_env)
    defensive_tool.conduct_attack(
        num_rounds=AttackConfig.TOTAL_ROUNDS,
        attack_start_round=AttackConfig.ATTACK_START_ROUND
    )

def main():
    """Run all examples"""
    print("🎯 Sybil Attack Tool - Example Usage")
    print("🚨 Educational and Research Purposes Only")
    print("=" * 70)
    
    # Run examples
    example_1_basic_attack()
    example_2_configurable_attack()
    example_3_comparison_study()
    example_4_custom_poisoning()
    example_5_defense_evaluation()
    
    print("\n🏁 All examples completed!")
    print("\nThese examples demonstrate:")
    print("• Basic sybil attack implementation")
    print("• Configurable attack parameters")
    print("• Attack intensity comparison")
    print("• Custom poisoning strategies")
    print("• Defense mechanism evaluation")
    
    print("\n⚠️  Important Notes:")
    print("• This tool is for educational and research purposes only")
    print("• Do not use for malicious activities")
    print("• Results may vary due to randomness")
    print("• Consider ethical implications in your research")

if __name__ == "__main__":
    main() 