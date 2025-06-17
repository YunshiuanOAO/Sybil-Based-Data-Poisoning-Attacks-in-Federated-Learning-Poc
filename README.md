# Sybil-Based Untargeted Data Poisoning Attack Tool

🚨 **Educational and Research Purposes Only** 🚨

This tool implements a sybil-based untargeted poisoning attack in federated learning scenarios. The attack creates multiple fake clients (sybil nodes) that inject poisoned data to degrade the overall model performance without targeting specific classes.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Attack Strategies](#attack-strategies)
- [Defense Mechanisms](#defense-mechanisms)
- [Output Analysis](#output-analysis)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

### What is a Sybil Attack?

A sybil attack in federated learning occurs when an adversary creates multiple fake identities (sybil nodes) to gain a disproportionately large influence on the global model. This tool demonstrates:

- **Untargeted Poisoning**: Degrading overall model performance rather than targeting specific classes
- **Federated Learning Simulation**: Complete FL environment with honest and malicious clients
- **Multiple Attack Strategies**: Label flipping, noise injection, and gradient manipulation
- **Comprehensive Analysis**: Detailed visualizations and metrics

### Key Components

- **FederatedLearningEnvironment**: Simulates FL with honest and sybil clients
- **SybilClient**: Implements malicious clients with poisoning capabilities
- **SybilAttackTool**: Orchestrates the attack and provides analysis
- **Configuration System**: Customizable attack parameters and scenarios

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone or download the project
# Navigate to the project directory

# Install required packages
pip install -r requirements.txt
```

### Required Packages

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## ⚡ Quick Start

### Basic Attack

```python
from sybil_attack_tool import *

# Create federated learning environment
fl_env = FederatedLearningEnvironment(
    num_honest_clients=5,
    num_sybil_clients=3,
    dataset_name='MNIST'
)

# Create and run attack
attack_tool = SybilAttackTool(fl_env)
attack_tool.conduct_attack(num_rounds=15, attack_start_round=5)
```

### Run Complete Demonstration

```bash
python sybil_attack_tool.py
```

### Run Example Scenarios

```bash
python example_usage.py
```

## 🎛️ Features

### Attack Capabilities

- **Multiple Sybil Clients**: Simulate various numbers of malicious clients
- **Untargeted Poisoning**: Degrade overall model performance
- **Label Flipping**: Randomly corrupt training labels
- **Noise Injection**: Add Gaussian noise to training data
- **Gradient Manipulation**: Use gradient ascent instead of descent
- **Timing Control**: Specify when attacks begin during training

### Analysis Tools

- **Real-time Monitoring**: Track accuracy and loss during training
- **Visual Analysis**: Comprehensive plots showing attack impact
- **Performance Metrics**: Detailed statistics on attack effectiveness
- **Export Results**: Save data and visualizations for further analysis

### Customization Options

- **Configurable Parameters**: Easy-to-modify attack settings
- **Multiple Scenarios**: Predefined attack intensities (mild, moderate, severe, stealth)
- **Custom Poisoning**: Extend with your own poisoning strategies
- **Defense Evaluation**: Test attacks against defensive mechanisms

## 📖 Usage Examples

### Example 1: Basic Attack

```python
# Simple sybil attack with default parameters
fl_env = FederatedLearningEnvironment(
    num_honest_clients=5,
    num_sybil_clients=3
)
attack_tool = SybilAttackTool(fl_env)
attack_tool.conduct_attack(num_rounds=10, attack_start_round=3)
```

### Example 2: Configurable Attack

```python
from config import AttackConfig, AttackScenarios

# Use predefined moderate attack scenario
AttackScenarios.moderate_attack()

fl_env = FederatedLearningEnvironment(
    num_honest_clients=AttackConfig.NUM_HONEST_CLIENTS,
    num_sybil_clients=AttackConfig.NUM_SYBIL_CLIENTS
)
attack_tool = SybilAttackTool(fl_env)
attack_tool.conduct_attack(
    num_rounds=AttackConfig.TOTAL_ROUNDS,
    attack_start_round=AttackConfig.ATTACK_START_ROUND
)
```

### Example 3: Custom Poisoning Strategy

```python
class CustomSybilClient(SybilClient):
    def _create_poisoned_dataset(self, original_dataset):
        # Implement your custom poisoning logic here
        # This example shows label flipping to adjacent classes
        # ... (see example_usage.py for full implementation)
        pass

# Use custom sybil clients
fl_env = FederatedLearningEnvironment(num_honest_clients=6, num_sybil_clients=2)
fl_env.sybil_clients = [CustomSybilClient(...) for i in range(2)]
```

## ⚙️ Configuration

### Attack Parameters

```python
class AttackConfig:
    # Environment
    NUM_HONEST_CLIENTS = 5      # Number of honest clients
    NUM_SYBIL_CLIENTS = 3       # Number of sybil clients
    DATASET_NAME = 'MNIST'      # Dataset to use
    
    # Attack Strategy
    POISON_RATIO = 0.3          # Ratio of poisoned samples
    ATTACK_START_ROUND = 5      # When to start the attack
    TOTAL_ROUNDS = 15           # Total training rounds
    
    # Training
    LEARNING_RATE_HONEST = 0.01 # LR for honest clients
    LEARNING_RATE_SYBIL = 0.05  # LR for sybil clients (higher for more damage)
    
    # Poisoning
    LABEL_FLIP_PROBABILITY = 0.3 # Probability of label flipping
    NOISE_INTENSITY = 0.1        # Intensity of noise injection
```

### Predefined Scenarios

```python
from config import AttackScenarios

# Configure different attack intensities
AttackScenarios.mild_attack()      # Low impact attack
AttackScenarios.moderate_attack()  # Balanced attack
AttackScenarios.severe_attack()    # High impact attack
AttackScenarios.stealth_attack()   # Hard-to-detect attack
```

## 🗡️ Attack Strategies

### 1. Label Flipping

Randomly changes training labels to incorrect values:

```python
# Random label flipping
poisoned_label = random.randint(0, num_classes - 1)

# Adjacent class flipping (more subtle)
poisoned_label = (original_label + 1) % num_classes
```

### 2. Noise Injection

Adds Gaussian noise to training data:

```python
noise = torch.randn_like(data) * noise_intensity
poisoned_data = torch.clamp(data + noise, 0, 1)
```

### 3. Gradient Manipulation

Uses gradient ascent instead of descent to maximize loss:

```python
# Maximize loss instead of minimizing (untargeted attack)
loss = -criterion(output, target)  # Negative loss
loss.backward()
optimizer.step()
```

### 4. Model Parameter Amplification

Amplifies malicious updates during aggregation:

```python
# Amplify sybil client contributions
amplified_params = sybil_params * amplification_factor
```

## 🛡️ Defense Mechanisms

The tool includes basic defense mechanisms for evaluation:

### Trimmed Mean Aggregation

```python
class DefensiveAggregation(SybilAttackTool):
    def _federated_averaging(self, local_models):
        # Calculate parameter distances from median
        # Remove outliers (potential sybil clients)
        # Aggregate only trusted models
        pass
```

### Parameter Distance Analysis

```python
# Calculate distances between model parameters
distances = [torch.norm(params - median_params) for params in all_params]

# Filter out models with high distances
trusted_models = filter_outliers(local_models, distances)
```

## 📊 Output Analysis

### Generated Files

- **`sybil_attack_analysis.png`**: Comprehensive visualization of attack results
- **`sybil_attack_results.csv`**: Detailed numerical results for further analysis

### Visualization Components

1. **Accuracy Over Time**: Shows model performance degradation
2. **Loss Over Time**: Displays training loss progression
3. **Attack Impact**: Quantifies accuracy degradation
4. **Attack Summary**: Configuration and effectiveness metrics

### Key Metrics

- **Maximum Accuracy Drop**: Peak performance degradation
- **Final Model Accuracy**: End-of-training performance
- **Attack Effectiveness**: Categorized as High/Moderate/Low
- **Sybil Ratio**: Proportion of malicious clients

## ⚖️ Ethical Considerations

### Research Ethics

This tool is designed for:

- ✅ **Educational purposes**: Understanding federated learning vulnerabilities
- ✅ **Security research**: Developing defense mechanisms
- ✅ **Academic studies**: Publishing research on FL security
- ✅ **Red team exercises**: Testing system robustness

### Prohibited Uses

- ❌ **Malicious attacks**: Do not use against real systems without permission
- ❌ **Unauthorized testing**: Only test on systems you own or have explicit permission
- ❌ **Commercial exploitation**: Do not use for unauthorized competitive advantage

### Best Practices

1. **Obtain Permission**: Always get explicit authorization before testing
2. **Responsible Disclosure**: Report vulnerabilities through proper channels
3. **Educational Focus**: Use for learning and improving security
4. **Documentation**: Maintain clear records of research activities

## 🚨 Disclaimer

**This tool is provided for educational and research purposes only.**

- The authors are not responsible for any misuse of this tool
- Users must comply with all applicable laws and regulations
- This tool should only be used in controlled, authorized environments
- Consider the ethical implications of your research

## 🤝 Contributing

We welcome contributions that improve the educational value and research capabilities of this tool:

### Areas for Contribution

- **New Attack Strategies**: Implement additional poisoning techniques
- **Defense Mechanisms**: Add more sophisticated defensive aggregation methods
- **Dataset Support**: Extend support to additional datasets (CIFAR-10, etc.)
- **Visualization**: Improve analysis and visualization capabilities
- **Documentation**: Enhance documentation and examples

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/abs/1803.07365)
- [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)

---

**Remember: Use this tool responsibly and ethically. The goal is to improve federated learning security, not to cause harm.** 