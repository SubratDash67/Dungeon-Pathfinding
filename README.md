# Real-Time Dungeon Pathfinding Optimizer using Graph Neural Networks

## 🚀 Overview
This project implements a real-time pathfinding system for procedurally generated dungeons using a custom Graph Neural Network (GNN) architecture. The system dynamically avoids threats (traps, guards) and predicts optimal paths in complex spatial environments.

---

## 🧠 Key Features

- **Procedural Dungeon Generation**: Built using cellular automata for rich, diverse layouts.
- **Dynamic Threat Simulation**: Includes trap triggering and patrol modeling.
- **Graph Neural Network Model**: Custom GNN layers optimized for spatial message passing.
- **High-Accuracy Path Prediction**: 90%+ accuracy with optimized F1 and recall.
- **Custom Optimization**: Loss functions and thresholds tuned for real-time use cases.

---

## 🛠 Technical Highlights

### 🔁 Dungeon Graph Construction
- Connects start to goal nodes in all cases.
- Graph nodes represent rooms; edges represent valid paths.
- Features encode room type, position, trap presence, guard proximity.

### 🧬 GNN Architecture
- Multi-layer spatial message-passing network with:
  - Residual connections
  - Batch normalization
  - Node-wise and edge-wise updates

### 📈 Training Strategy
- Recall-optimized custom loss function.
- Threshold tuning for optimal F1 score.
- Heavy data augmentation on dungeon layouts and threats.
- Evaluation via precision, recall, F1, and accuracy.

---

## 📁 Project Structure

```
RealTimeDungeonPathfindingOptimizer/
├── data/
│   ├── dungeon_graphs.py
│   ├── trap_guard_simulation.py
│   └── dataset_generator.py
├── models/
│   ├── gnn_model.py
│   └── gnn_layers.py
├── training/
│   ├── trainer.py
│   └── checkpoint.py
├── utils/
│   ├── data_augmentation.py
│   ├── graph_utils.py
│   ├── logging_utils.py
│   └── custom_loss.py
├── scripts/
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── enhanced_training.py
│   └── enhanced_evaluation.py
└── tests/
```

---

## 📊 Performance Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 90.5%  |
| Precision | 84.3%  |
| Recall    | 79.8%  |
| F1 Score  | 82.0%  |

---

## ⚙️ Usage

### 🔧 Environment Setup

```
python -m venv venv
# On Unix/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate

pip install torch torch-geometric numpy matplotlib tqdm
```

### 🏃‍♂️ Run Training

```
python scripts/enhanced_training.py
```

### 📈 Evaluate Model

```
python scripts/enhanced_evaluation.py \
  --checkpoint checkpoints/enhanced_training_[TIMESTAMP]/best_checkpoint.pt \
  --num_test_samples 5
```

---

## 🔭 Future Work

- Integrate real-time planner into game engines (Unity/Unreal)
- Reinforcement Learning for adaptive navigation
- Support multi-agent cooperative pathfinding
- Visual interactive debugger for path interpretation

---

## 🧰 Technologies Used

- **Python 3.8+**
- **PyTorch**, **PyTorch Geometric**
- **NetworkX**, **NumPy**, **Matplotlib**
