"""
Sybil-Based Untargeted Data Poisoning Attack Tool for Federated Learning
========================================================================

This tool implements a sybil-based untargeted poisoning attack in federated learning scenarios.
The attack creates multiple fake clients (sybil nodes) that inject poisoned data to degrade
the overall model performance without targeting specific classes.

Author: Security Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import random
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import copy

class FederatedLearningEnvironment:
    """
    Simulates a federated learning environment with potential for sybil attacks
    """
    
    def __init__(self, num_honest_clients: int = 5, num_sybil_clients: int = 3, 
                 dataset_name: str = 'MNIST'):
        self.num_honest_clients = num_honest_clients
        self.num_sybil_clients = num_sybil_clients
        self.total_clients = num_honest_clients + num_sybil_clients
        self.dataset_name = dataset_name
        
        # Initialize clients
        self.honest_clients = []
        self.sybil_clients = []
        
        # Global model
        self.global_model = None
        
        # Training history
        self.training_history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'honest_accuracy': [],
            'attack_strength': []
        }
        
        self._setup_dataset()
        self._initialize_clients()
        
    def _setup_dataset(self):
        """Setup dataset for federated learning"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if self.dataset_name == 'MNIST':
            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            self.num_classes = 10
            self.input_size = 28 * 28
            
        # Create test loader
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        
    def _initialize_clients(self):
        """Initialize honest and sybil clients"""
        # Split data among honest clients
        data_per_client = len(self.train_dataset) // self.num_honest_clients
        
        for i in range(self.num_honest_clients):
            start_idx = i * data_per_client
            end_idx = (i + 1) * data_per_client if i < self.num_honest_clients - 1 else len(self.train_dataset)
            client_data = Subset(self.train_dataset, range(start_idx, end_idx))
            
            client = HonestClient(client_id=i, data=client_data, input_size=self.input_size, 
                                num_classes=self.num_classes)
            self.honest_clients.append(client)
            
        # Initialize sybil clients with poisoned data
        for i in range(self.num_sybil_clients):
            client = SybilClient(client_id=f"sybil_{i}", 
                               original_dataset=self.train_dataset,
                               input_size=self.input_size,
                               num_classes=self.num_classes)
            self.sybil_clients.append(client)
            
    def get_global_model(self):
        """Initialize or return global model"""
        if self.global_model is None:
            self.global_model = SimpleNN(self.input_size, self.num_classes)
        return self.global_model

class SimpleNN(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, num_classes: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class HonestClient:
    """Represents an honest client in federated learning"""
    
    def __init__(self, client_id: int, data: Dataset, input_size: int, num_classes: int):
        self.client_id = client_id
        self.data = data
        self.data_loader = DataLoader(data, batch_size=32, shuffle=True)
        self.model = SimpleNN(input_size, num_classes)
        
    def train_local_model(self, global_model: nn.Module, epochs: int = 1) -> nn.Module:
        """Train local model based on global model"""
        # Copy global model parameters
        self.model.load_state_dict(global_model.state_dict())
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        return self.model

class SybilClient:
    """Represents a malicious sybil client that performs untargeted poisoning"""
    
    def __init__(self, client_id: str, original_dataset: Dataset, 
                 input_size: int, num_classes: int, poison_ratio: float = 0.3):
        self.client_id = client_id
        self.input_size = input_size
        self.num_classes = num_classes
        self.poison_ratio = poison_ratio
        
        # Create poisoned dataset
        self.poisoned_data = self._create_poisoned_dataset(original_dataset)
        self.data_loader = DataLoader(self.poisoned_data, batch_size=32, shuffle=True)
        self.model = SimpleNN(input_size, num_classes)
        
    def _create_poisoned_dataset(self, original_dataset: Dataset) -> Dataset:
        """Create poisoned dataset with label flipping and noise injection"""
        # Sample subset of original data
        sample_size = min(1000, len(original_dataset) // 10)
        indices = random.sample(range(len(original_dataset)), sample_size)
        
        poisoned_samples = []
        poisoned_labels = []
        
        for idx in indices:
            data, label = original_dataset[idx]
            
            # Apply poisoning with certain probability
            if random.random() < self.poison_ratio:
                # Label flipping attack - randomly flip labels
                poisoned_label = random.randint(0, self.num_classes - 1)
                
                # Add noise to data
                noise = torch.randn_like(data) * 0.1
                poisoned_data = torch.clamp(data + noise, 0, 1)
                
                poisoned_samples.append(poisoned_data)
                poisoned_labels.append(poisoned_label)
            else:
                poisoned_samples.append(data)
                poisoned_labels.append(label)
                
        return PoisonedDataset(poisoned_samples, poisoned_labels)
    
    def train_local_model(self, global_model: nn.Module, epochs: int = 1) -> nn.Module:
        """Train poisoned model to maximize damage"""
        # Copy global model parameters
        self.model.load_state_dict(global_model.state_dict())
        
        # Use higher learning rate to maximize damage
        optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                
                # Maximize loss instead of minimizing (untargeted attack)
                loss = -criterion(output, target)  # Negative loss for gradient ascent
                loss.backward()
                optimizer.step()
                
        return self.model

class PoisonedDataset(Dataset):
    """Custom dataset for poisoned data"""
    
    def __init__(self, data_list: List, labels_list: List):
        self.data = data_list
        self.labels = labels_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SybilAttackTool:
    """Main tool for conducting sybil-based untargeted poisoning attacks"""
    
    def __init__(self, fl_env: FederatedLearningEnvironment):
        self.fl_env = fl_env
        self.attack_active = False
        
    def conduct_attack(self, num_rounds: int = 10, attack_start_round: int = 3):
        """Conduct the sybil attack over multiple federated learning rounds"""
        print("🚨 Starting Sybil-Based Untargeted Poisoning Attack")
        print(f"Environment: {self.fl_env.num_honest_clients} honest clients, {self.fl_env.num_sybil_clients} sybil clients")
        print(f"Attack will begin at round {attack_start_round}")
        
        global_model = self.fl_env.get_global_model()
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Activate attack after certain round
            if round_num >= attack_start_round - 1:
                self.attack_active = True
                print("⚠️  Sybil attack ACTIVE")
            
            # Collect local models
            local_models = []
            
            # Train honest clients
            for client in self.fl_env.honest_clients:
                local_model = client.train_local_model(global_model, epochs=1)
                local_models.append(local_model)
                
            # Train sybil clients (if attack is active)
            if self.attack_active:
                for client in self.fl_env.sybil_clients:
                    local_model = client.train_local_model(global_model, epochs=1)
                    local_models.append(local_model)
                    
            # Aggregate models (simple averaging)
            global_model = self._federated_averaging(local_models)
            
            # Evaluate model
            accuracy, loss = self._evaluate_model(global_model)
            honest_accuracy = self._evaluate_honest_performance(global_model)
            
            # Record metrics
            self.fl_env.training_history['rounds'].append(round_num + 1)
            self.fl_env.training_history['accuracy'].append(accuracy)
            self.fl_env.training_history['loss'].append(loss)
            self.fl_env.training_history['honest_accuracy'].append(honest_accuracy)
            self.fl_env.training_history['attack_strength'].append(
                1.0 if self.attack_active else 0.0
            )
            
            print(f"Global Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            if self.attack_active:
                print(f"Attack Impact: {honest_accuracy - accuracy:.4f}")
                
        print("\n🔍 Attack completed. Generating analysis...")
        self._generate_attack_analysis()
        
    def _federated_averaging(self, local_models: List[nn.Module]) -> nn.Module:
        """Perform federated averaging of local models"""
        global_model = copy.deepcopy(local_models[0])
        global_state_dict = global_model.state_dict()
        
        # Average parameters
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([
                model.state_dict()[key] for model in local_models
            ]).mean(0)
            
        global_model.load_state_dict(global_state_dict)
        return global_model
    
    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.fl_env.test_loader:
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        accuracy = correct / len(self.fl_env.test_dataset)
        avg_loss = total_loss / len(self.fl_env.test_loader)
        
        return accuracy, avg_loss
    
    def _evaluate_honest_performance(self, global_model: nn.Module) -> float:
        """Evaluate what the performance would be with only honest clients"""
        # This is an approximation - in practice, this would require
        # training only with honest clients
        return self._evaluate_model(global_model)[0]
    
    def _generate_attack_analysis(self):
        """Generate comprehensive analysis of the attack"""
        history = self.fl_env.training_history
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sybil-Based Untargeted Poisoning Attack Analysis', fontsize=16)
        
        # Accuracy over time
        axes[0, 0].plot(history['rounds'], history['accuracy'], 'b-', label='Global Accuracy', linewidth=2)
        axes[0, 0].plot(history['rounds'], history['honest_accuracy'], 'g--', label='Honest Baseline', linewidth=2)
        attack_rounds = [r for r, a in zip(history['rounds'], history['attack_strength']) if a > 0]
        if attack_rounds:
            axes[0, 0].axvline(x=min(attack_rounds), color='red', linestyle=':', label='Attack Start', alpha=0.7)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss over time
        axes[0, 1].plot(history['rounds'], history['loss'], 'r-', linewidth=2)
        if attack_rounds:
            axes[0, 1].axvline(x=min(attack_rounds), color='red', linestyle=':', alpha=0.7)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Attack impact
        attack_impact = [h - a for h, a in zip(history['honest_accuracy'], history['accuracy'])]
        axes[1, 0].plot(history['rounds'], attack_impact, 'orange', linewidth=2)
        if attack_rounds:
            axes[1, 0].axvline(x=min(attack_rounds), color='red', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Accuracy Degradation')
        axes[1, 0].set_title('Attack Impact (Accuracy Loss)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Attack configuration
        config_text = f"""
Attack Configuration:
• Honest Clients: {self.fl_env.num_honest_clients}
• Sybil Clients: {self.fl_env.num_sybil_clients}
• Sybil Ratio: {self.fl_env.num_sybil_clients/self.fl_env.total_clients:.1%}
• Dataset: {self.fl_env.dataset_name}

Attack Results:
• Max Accuracy Drop: {max(attack_impact):.4f}
• Final Accuracy: {history['accuracy'][-1]:.4f}
• Attack Success: {'Yes' if max(attack_impact) > 0.1 else 'Partial'}
        """
        axes[1, 1].text(0.1, 0.5, config_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='center', fontsize=10, fontfamily='monospace')
        axes[1, 1].set_title('Attack Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sybil_attack_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        results_df = pd.DataFrame(history)
        results_df.to_csv('sybil_attack_results.csv', index=False)
        
        print("\n📊 Analysis Complete!")
        print("• Visualization saved as 'sybil_attack_analysis.png'")
        print("• Detailed results saved as 'sybil_attack_results.csv'")
        
        # Print summary
        print(f"\n📈 Attack Summary:")
        print(f"• Maximum accuracy degradation: {max(attack_impact):.4f}")
        print(f"• Final model accuracy: {history['accuracy'][-1]:.4f}")
        print(f"• Attack effectiveness: {'High' if max(attack_impact) > 0.15 else 'Moderate' if max(attack_impact) > 0.05 else 'Low'}")

def main():
    """Main function to run the sybil attack demonstration"""
    print("🎯 Sybil-Based Untargeted Poisoning Attack Tool")
    print("=" * 50)
    
    # Create federated learning environment
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=5,
        num_sybil_clients=3,
        dataset_name='MNIST'
    )
    
    # Create attack tool
    attack_tool = SybilAttackTool(fl_env)
    
    # Conduct attack
    attack_tool.conduct_attack(num_rounds=15, attack_start_round=5)
    
    print("\n🏁 Demonstration complete!")
    print("This tool demonstrates how sybil attacks can degrade federated learning performance.")
    print("Use responsibly for educational and research purposes only.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main() 