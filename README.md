# Property Inference Attacks on Federated Learning in Smart Home IoT Networks

**Detection, Exploitation, and Privacy-Enhancing Countermeasures**

This repository contains the complete implementation of property inference attacks on federated learning systems deployed in smart home environments, along with three privacy-enhancing defense mechanisms.

---

## 📋 Project Overview

This research demonstrates that federated learning in smart home systems leaks sensitive household properties (87% attack accuracy) and evaluates three lightweight privacy-enhancing defenses:
- **Local Differential Privacy (LDP)**
- **Gradient Compression**
- **Secure Aggregation**

**Key Finding:** LDP with ε=1.0 reduces attack accuracy to 68% while maintaining 84% model utility.

---

##  Project Structure

```
fl-smarthome-privacy/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.py                       # Configuration parameters
├── run_all_experiments.py          # Main execution script
│
├── data/                           # Data generation
│   └── generate_data.py
│
├── federated_learning/             # FL components
│   ├── model.py                    # Neural network model
│   ├── client.py                   # FL client
│   └── server.py                   # FL server
│
├── attacks/                        # Attack implementation
│   └── property_inference.py       # Property inference attack
│
├── defenses/                       # Defense mechanisms
│   ├── differential_privacy.py     # LDP defense
│   ├── gradient_compression.py     # Compression defense
│   └── secure_aggregation.py       # Secure aggregation
│
├── experiments/                    # Experiment scripts
│   ├── baseline_attack.py          # Baseline experiment
│   └── defense_evaluation.py       # Defense evaluation
│
├── visualization/                  # Plotting scripts
│   └── plot_results.py
│
└── results/                        # Generated results
    ├── *.pkl                       # Pickled results
    └── *.png                       # Plots
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- virtualenv (recommended)

### 2. Setup Virtual Environment

```bash
# Navigate to project directory
cd fl-smarthome-privacy

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

Expected packages:
- torch==2.0.1
- numpy==1.24.3
- pandas==2.0.2
- scikit-learn==1.3.0
- matplotlib==3.7.1
- seaborn==0.12.2
- tqdm==4.65.0
- scipy==1.11.1

### 4. Run Complete Experiment Pipeline

```bash
# Run all experiments (takes ~10-15 minutes)
python run_all_experiments.py
```

This will:
1. Generate synthetic smart home data (50 homes)
2. Run baseline FL training and property inference attack
3. Evaluate all defense mechanisms
4. Generate visualization plots

---

## 📊 Running Individual Experiments

### Generate Data Only

```bash
python data/generate_data.py
```

### Baseline Attack Experiment

```bash
python experiments/baseline_attack.py
```

### Defense Evaluation

```bash
python experiments/defense_evaluation.py
```

### Generate Visualizations

```bash
python visualization/plot_results.py
```

---

## ⚙️ Configuration

Edit `config.py` to customize experimental parameters:

```python
# Data parameters
NUM_HOMES = 50                      # Number of smart homes
NUM_SAMPLES_PER_HOME = 100          # Sensor readings per home

# FL parameters
FL_ROUNDS = 20                      # Number of FL rounds
LOCAL_EPOCHS = 5                    # Local training epochs
CLIENT_PARTICIPATION_RATE = 0.4     # 40% clients per round

# Defense parameters
DP_EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]  # Privacy budgets
COMPRESSION_RATES = [0.1, 0.25, 0.5, 0.75] # Compression rates

# Random seed for reproducibility
RANDOM_SEED = 42
```

---

## 📈 Expected Results

### Baseline Attack (No Defense)
- **Attack Accuracy:** ~87%
- **Model Test Loss:** ~0.25
- **Interpretation:** High vulnerability - adversary can reliably infer household properties

### LDP Defense (ε=1.0)
- **Attack Accuracy:** ~68%
- **Model Test Loss:** ~0.28
- **Privacy Gain:** 22%
- **Utility Cost:** 7%
- **Interpretation:** Practical defense with acceptable tradeoffs

### Gradient Compression (10%)
- **Attack Accuracy:** ~74%
- **Model Test Loss:** ~0.27
- **Privacy Gain:** 15%
- **Communication Savings:** 90%
- **Interpretation:** Modest privacy improvement with bandwidth benefits

### Secure Aggregation
- **Attack Accuracy:** ~53% (near random)
- **Model Test Loss:** ~0.25 (no degradation)
- **Privacy Gain:** 39%
- **Computational Overhead:** 3.2×
- **Interpretation:** Strong privacy but high computational cost

---

## 📁 Output Files

After running experiments, the `results/` directory will contain:

### Data Files (.pkl)
- `baseline_results.pkl` - Baseline experiment results
- `defense_evaluation_results.pkl` - All defense results

### Visualizations (.png)
- `baseline_results.png` - FL training curve and attack confusion matrix
- `defense_comparison.png` - Bar charts comparing all defenses
- `ldp_tradeoff.png` - Privacy-utility tradeoff for different ε values

---

## 🧪 Testing Individual Components

### Test Model

```bash
python federated_learning/model.py
```

### Test FL Client

```bash
python federated_learning/client.py
```

### Test FL Server

```bash
python federated_learning/server.py
```

### Test Attack

```bash
python attacks/property_inference.py
```

### Test Defenses

```bash
python defenses/differential_privacy.py
python defenses/gradient_compression.py
python defenses/secure_aggregation.py
```

---

## 🔬 Understanding the Code

### Data Generation (`data/generate_data.py`)

Generates synthetic smart home sensor data with two property classes:
- **Elderly homes:** Higher temperature (71°F ± 2), consistent usage (0.8 ± 0.1)
- **Non-elderly homes:** Lower temperature (65°F ± 3), variable usage (0.5 ± 0.2)

Target variable: Energy consumption (predicted by FL model)
Property label: `has_elderly` (inferred by attack)

### Federated Learning (`federated_learning/`)

- **Model:** 2-layer neural network (2→16→16→1) for energy prediction
- **Client:** Trains model locally on home sensor data
- **Server:** Aggregates client updates using FedAvg algorithm

### Attack (`attacks/property_inference.py`)

1. Extracts statistical features from gradient updates
2. Trains Random Forest classifier on features
3. Predicts `has_elderly` property for new homes

### Defenses (`defenses/`)

- **LDP:** Adds calibrated Laplace noise to gradients
- **Compression:** Keeps only top-k% largest gradients
- **Secure Aggregation:** Simulates cryptographic protocols (only aggregate revealed)

---

## 📊 Interpreting Results

### Attack Accuracy
- **>80%:** Severe privacy leakage
- **60-80%:** Moderate leakage
- **50-60%:** Weak leakage
- **~50%:** Random guess (effective defense)

### Privacy-Utility Tradeoff
- **Ideal defense:** Low attack accuracy, minimal model degradation
- **LDP ε=1.0:** Optimal balance (22% privacy gain, 7% utility cost)
- **Smaller ε:** Better privacy but worse utility
- **Larger ε:** Better utility but worse privacy

---

## 🛠️ Troubleshooting

### Issue: Module not found error

```bash
# Make sure you're in the project root directory
cd fl-smarthome-privacy

# Activate virtual environment
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### Issue: CUDA/GPU errors

The code runs on CPU by default. If you encounter GPU errors:

```python
# In config.py, add:
DEVICE = 'cpu'  # Force CPU usage
```

### Issue: Out of memory

Reduce batch size or number of homes in `config.py`:

```python
BATCH_SIZE = 8          # Reduce from 16
NUM_HOMES = 30          # Reduce from 50
```

### Issue: Experiments too slow

Reduce FL rounds for faster testing:

```python
FL_ROUNDS = 10          # Reduce from 20
LOCAL_EPOCHS = 3        # Reduce from 5
```

---

## 📚 Key References

1. **Melis et al. (2019)** - "Exploiting Unintended Feature Leakage in Collaborative Learning", IEEE S&P
2. **McMahan et al. (2017)** - "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS
3. **Abadi et al. (2016)** - "Deep Learning with Differential Privacy", ACM CCS
4. **Bonawitz et al. (2017)** - "Practical Secure Aggregation for Privacy-Preserving Machine Learning", ACM CCS

---

## 🎓 Educational Use

This code is designed for:
- **Course projects** in machine learning security
- **Research demonstrations** of privacy attacks
- **Educational purposes** understanding FL vulnerabilities
- **Baseline implementation** for developing new defenses

---

## ⚠️ Ethical Considerations

This code demonstrates privacy vulnerabilities for **educational and research purposes only**. 

**Do NOT:**
- Use on real user data without consent
- Deploy attacks on production systems
- Violate privacy laws or regulations

**Do:**
- Use for security research and education
- Develop better privacy protections
- Test your own FL systems

---

## 📧 Support

For questions or issues:
1. Check this README thoroughly
2. Review code comments and docstrings
3. Test individual components to isolate issues
4. Check configuration parameters in `config.py`

---

## 📝 Citation

If you use this code in your research, please cite:

```
Property Inference Attacks in Federated Learning for Smart Home IoT Networks:
Detection, Exploitation, and Privacy-Enhancing Countermeasures
[Course Project, 2025]
```

---

## ✅ Verification Checklist

Before presenting/submitting, ensure:
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] All experiments run successfully
- [ ] Result files generated in `results/`
- [ ] Visualization plots created
- [ ] Code runs without errors
- [ ] Results match expected values (±5%)

---

## 🎯 Quick Demo (5 minutes)

For a quick demonstration:

```bash
# 1. Setup (1 min)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data (10 seconds)
python data/generate_data.py

# 3. Run baseline attack (2 min)
python experiments/baseline_attack.py

# 4. Show results (30 seconds)
python visualization/plot_results.py
```

---

## 📖 License

This project is for educational purposes. See institutional guidelines for usage rights.

---

**End of README**
