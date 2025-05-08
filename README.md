# Real-Time Dungeon Pathfinding Optimizer using Graph Neural Networks

## 🚀 Overview
This project implements a real-time pathfinding system for procedurally generated dungeons using a custom Graph Neural Network (GNN) architecture. The system dynamically avoids threats (traps, guards) and predicts optimal paths in complex spatial environments.

![Dungeon Pathfinding Example](evaluation_results/new_dungeons_eval/enhanced_prediction_1.png)

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

