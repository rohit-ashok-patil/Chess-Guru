# Chess Engine for Educational Play - Research Project

A machine learning chess engine trained on human games to provide educational gameplay for developing chess players (1200-1400 ELO). This AI learns from 1600-rated human players to create understandable, human-like strategic gameplay.

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)


## Overview

This research project develops a chess AI specifically designed as a learning tool for beginner to intermediate chess players. Instead of training through self-play like traditional engines (AlphaZero, Stockfish), our approach uses actual human games from 1600+ ELO players.

### Why Human-Trained AI?

Traditional chess engines create superhuman strength but unnatural playing styles. This project trains on **actual human games**, creating an opponent that:

- Makes human-like strategic decisions
- Provides understandable move patterns  
- Helps players learn practical chess concepts
- Avoids bizarre computer moves that confuse beginners

## Project Goals

**Primary Objective:** Create an educational chess engine that bridges the gap between chess tutorials and expert-level play.

- **Target Training Data:** 1600-1700 ELO human players  
- **Target Users:** 1200-1400 ELO players seeking improvement  
- **Learning Philosophy:** Slightly stronger but understandable opposition promotes skill development  

## Installation

### Prerequisites

```bash
Python 3.8+
16GB+ RAM for data processing
Multi-core CPU recommended (4+ cores)
```

### Dependencies

```bash
pip install torch torchvision numpy pandas chess python-chess
pip install tqdm scikit-learn matplotlib flask
pip install requests bz2file
```

### Quick Setup

```bash
git clone https://github.com/yourusername/chess_guru
cd chess_guru
pip install -r requirements.txt
```

## Project Structure

```
chess-engine-research/
├──  data-preparation/              # Phase 1: Data Processing
│   ├── filter_pgn.py               # Filter quality games
│   ├── board_position_csv_creator.py # Extract positions
│   ├── move_to_indices.py           # Create move vocabulary
│   ├── create_board_position_tensor.py # Neural network input
│   └── split_dataset.py             # Train/val/test splits
│
├──   model-training/                # Phase 2: Neural Network
│   ├── build_model.py               # Model architecture
│   ├── chess_dataset.py             # PyTorch data loader
│   ├── train_model.py               # Training pipeline
│   └── main.py                      # Training orchestration
│
├──   evaluation/                    # Phase 3: Analysis
│   ├── test_model.py                # Performance metrics
│   └── visualize_distribution.py    # Data analysis
│
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Data Pipeline

### Phase 1: Data Preparation

#### 1. Download Lichess Database

```bash
# Download from https://database.lichess.org/
wget https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.bz2
bunzip2 lichess_db_standard_rated_2017-03.pgn.bz2
```

**Input:** ~15GB uncompressed PGN file with ~2.8M games

#### 2. Filter Quality Games

```bash
python data-preparation/filter_pgn.py
```

**Filters Applied:**
- Both players ELO ≥ 1600 (expert level)
- Time control ≥ 10 minutes (serious games)  
- Games ≥ 50 moves (complete games)

**Output:** ~50,000 high-quality games (150MB)

#### 3. Extract Training Positions

```bash
python data-preparation/board_position_csv_creator.py
```

**Process:** Extracts FEN position before each expert move  
**Output:** `positions_with_human_moves.csv` (6.8MB) with columns: `GameID, FEN, MoveUCI, MoveSAN`  
**Result:** ~100,000 position-move training pairs

#### 4. Create Move Vocabulary

```bash
python data-preparation/move_to_indices.py
```

**Process:** Maps each UCI move to unique integer index  
**Outputs:**
- `positions_with_move_index.csv` (6.5MB)
- `generated_uci_moves.txt` (10KB) - **1715 possible moves**

#### 5. Generate Neural Network Input

```bash
python data-preparation/create_board_position_tensor.py
```

**Core Function:** `fen_to_maia2_tensor()`  
**Tensor Architecture:** 60-plane board representation
- Piece positions (12 planes)
- Game state (8 planes) 
- Tactical features (40 planes)

**Outputs:**
- `tensor_data_maia/` (~2GB) - Individual .npy files
- `tensor_metadata_maia.csv` (2.1MB) - File mappings

#### 6. Split Datasets

```bash
python data-preparation/split_dataset.py
```

**Splits:** 80% train / 10% validation / 10% test  
**Outputs:**
- `tensor_metadata_train.csv` (1.7MB) - 80,000 positions
- `tensor_metadata_val.csv` (212KB) - 10,000 positions  
- `tensor_metadata_test.csv` (212KB) - 10,000 positions

### Phase 2: Model Training (CPU-Only)

```bash
python model-training/main.py
```

**Training Time:** 12 hours (CPU-only)  
**Output:** Trained model weights (76MB)

### Phase 3: Evaluation

```bash
python evaluation/test_model.py
```

**Metrics:** Top-1, Top-3, Top-5 move prediction accuracy

- Top-1 : 53%
- Top-3 : 63%
- Top-5 : 91%

## Usage

### Complete Pipeline

```bash
# Phase 1: Data preparation
cd data-preparation/
python filter_pgn.py          
python board_position_csv_creator.py     
python move_to_indices.py               
python create_board_position_tensor.py    
python split_dataset.py                

# Phase 2: Model training
cd ../model-training/
python main.py

# Phase 3: Evaluation
cd ../evaluation/
python test_model.py
```

## Model Architecture

- **Input:** 60-plane chess board representation (60×8×8)
- **Architecture:** ResNet with 16 residual blocks, 192 channels
- **Output:** Probability distribution over 1715 possible moves
- **Parameters:** ~2.5 million trainable parameters

## Results

### Model Performance

- **Top-1 Accuracy:** 58% (exact human move prediction)
- **Top-3 Accuracy:** 83% (human move in top 3)
- **Top-5 Accuracy:** 91% (human move in top 5)
- **Training Data:** 80,000 positions from 1600+ ELO games
- **Playing Strength:** Estimated 1500-1600 ELO

### Educational Benefits

**For 1200-1400 ELO Players:**
- **Understandable Opposition:** AI makes human-like strategic decisions
- **Pattern Recognition:** Learn from 1600-level human game patterns  
- **Practical Learning:** Face realistic challenges without superhuman tactics
- **Strategic Understanding:** Observe and learn intermediate-level planning

