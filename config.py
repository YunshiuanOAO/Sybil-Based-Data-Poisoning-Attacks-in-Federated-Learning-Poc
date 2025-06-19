"""
Configuration file for Sybil-Based Untargeted Poisoning Attack
==============================================================

This file contains configurable parameters for the attack simulation.
Modify these settings to experiment with different attack scenarios.
"""

class AttackConfig:
    """Configuration parameters for the sybil attack"""
    
    # Federated Learning Environment
    NUM_HONEST_CLIENTS = 5
    NUM_SYBIL_CLIENTS = 3
    DATASET_NAME = 'MNIST'  # Currently supports 'MNIST'
    
    # Attack Parameters
    POISON_RATIO = 0.3  # Ratio of poisoned samples in sybil client data
    ATTACK_START_ROUND = 5  # Round when sybil attack begins
    TOTAL_ROUNDS = 15  # Total number of federated learning rounds
    
    # Training Parameters
    LEARNING_RATE_HONEST = 0.01  # Learning rate for honest clients
    LEARNING_RATE_SYBIL = 0.05   # Learning rate for sybil clients (higher for more damage)
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 1  # Number of local training epochs per round
    
    # Model Parameters
    HIDDEN_SIZE_1 = 128
    HIDDEN_SIZE_2 = 64
    DROPOUT_RATE = 0.2
    
    # Poisoning Strategies
    LABEL_FLIP_PROBABILITY = 0.3  # Probability of label flipping
    NOISE_INTENSITY = 0.1  # Intensity of noise injection
    
    # Output Settings
    SAVE_RESULTS = True
    SAVE_PLOTS = True
    VERBOSE = True
    
    # Advanced Attack Settings
    GRADIENT_ASCENT = True  # Use gradient ascent instead of descent for sybil clients
    AMPLIFICATION_FACTOR = 2.0  # Amplify malicious updates
    
    @classmethod
    def get_attack_description(cls):
        """Return a description of the current attack configuration"""
        return f"""
Current Attack Configuration:
============================
Environment:
  - Honest Clients: {cls.NUM_HONEST_CLIENTS}
  - Sybil Clients: {cls.NUM_SYBIL_CLIENTS}
  - Sybil Ratio: {cls.NUM_SYBIL_CLIENTS/(cls.NUM_HONEST_CLIENTS + cls.NUM_SYBIL_CLIENTS):.1%}
  - Dataset: {cls.DATASET_NAME}

Attack Strategy:
  - Poison Ratio: {cls.POISON_RATIO:.1%}
  - Attack Start Round: {cls.ATTACK_START_ROUND}
  - Total Rounds: {cls.TOTAL_ROUNDS}
  - Label Flip Probability: {cls.LABEL_FLIP_PROBABILITY:.1%}
  - Noise Intensity: {cls.NOISE_INTENSITY}

Training:
  - Honest LR: {cls.LEARNING_RATE_HONEST}
  - Sybil LR: {cls.LEARNING_RATE_SYBIL}
  - Batch Size: {cls.BATCH_SIZE}
  - Local Epochs: {cls.LOCAL_EPOCHS}
        """

# Predefined attack scenarios
class AttackScenarios:
    """Predefined attack scenarios for easy testing"""
    
    @staticmethod
    def mild_attack():
        """Configure a mild sybil attack"""
        AttackConfig.NUM_SYBIL_CLIENTS = 2
        AttackConfig.POISON_RATIO = 0.2
        AttackConfig.LEARNING_RATE_SYBIL = 0.02
        AttackConfig.NOISE_INTENSITY = 0.05
        
    @staticmethod
    def moderate_attack():
        """Configure a moderate sybil attack"""
        AttackConfig.NUM_SYBIL_CLIENTS = 3
        AttackConfig.POISON_RATIO = 0.3
        AttackConfig.LEARNING_RATE_SYBIL = 0.05
        AttackConfig.NOISE_INTENSITY = 0.1
        
    @staticmethod
    def severe_attack():
        """Configure a severe sybil attack"""
        AttackConfig.NUM_SYBIL_CLIENTS = 7  # 更多惡意客戶端
        AttackConfig.POISON_RATIO = 0.6
        AttackConfig.LEARNING_RATE_SYBIL = 0.15
        AttackConfig.NOISE_INTENSITY = 0.3
        AttackConfig.AMPLIFICATION_FACTOR = 4.0  # 更強的放大係數
        
    @staticmethod
    def stealth_attack():
        """Configure a stealth sybil attack (harder to detect)"""
        AttackConfig.NUM_SYBIL_CLIENTS = 2
        AttackConfig.POISON_RATIO = 0.15
        AttackConfig.LEARNING_RATE_SYBIL = 0.015
        AttackConfig.NOISE_INTENSITY = 0.03
        AttackConfig.ATTACK_START_ROUND = 8 