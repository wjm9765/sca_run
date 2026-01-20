# Qwen3-Omni Full Duplex Runner

This repository contains the runner code for the **Qwen3-Omni** model fine-tuned for **Full Duplex** interaction. It allows for real-time voice conversations where the model listens and speaks simultaneously, enabling a fluid conversational experience.

## Training & Data Preparation

- **Data Preparation**: [sca_data_prep](https://github.com/riverfog7/sca_data_prep) - Refer to this repository for preparing the dataset.
- **Training**: [SCA](https://github.com/wjm9765/SCA) - Refer to this repository for the code used to train the model.

##  Features

- **Full Duplex Interaction**: Supports simultaneous listening and speaking functionality (interruptibility).
- **Streaming Pipeline**: Handles real-time audio streaming via WebSocket.
- **Modular Architecture**: Separates server logic, feature extraction, and model inference.
- **Easy Deployment**: Simple command-line interface for running the inference server.

## Configuration

The configuration is split into two main files depending on what you need to adjust:

### 1. Model & Core Parameters (`src/sca_run/config.py`)
Modify this file if you need to change the fundamental behavior of the model or audio specifications, such as:
- **Audio Sample Rate**
- **Number of tokens to predict** per step

### 2. Server & Environment (`config/default.toml`)
Modify this file for deployment-specific settings:
- **Model Path** (`model_id`)
- **Server Audio Processing** (buffering, chunking)

### Default Values
The system is pre-configured with the following defaults for optimal performance:
- **Audio Input**: 8 tokens (approx. 0.64s)
- **Text Generation**: 4 tokens
- **Talker Prediction**: approx. 0.64s

##  Getting Started

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (Recommended for Qwen3-Omni)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sca_run.git
   cd sca_run
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync --extra full,cu128 (cuda version depends on your GPU, for CPU only, use the cpu extras) to install the required dependencies.
   ```

### Running the Server

1. **Set Model Path** (Optional, if not set in `default.toml`):
   ```bash
   export SCA_QWEN_MODEL_ID="/path/to/your/finetuned/model"
   ```

2. **Start the Server**:
   ```bash
   ./scripts/run_demo_server.sh
   ```
   
   The server will start at `http://0.0.0.0:8000`.

3. **Web Interface**:
   Open a browser and navigate to `http://localhost:8000`. You can interact with the model using your microphone.

##  Project Structure

- `src/sca_run/`: Main source code
  - `server.py`: FastAPI server & WebSocket handler
  - `team_infer.py,inference.py`: Integration layer for Qwen3-Omni logic
  - `config.py`: Configuration dataclasses
  - `static/`: Frontend assets (Web UI)
- `config/`: Configuration files (`default.toml`)
- `scripts/`: Utility scripts for testing and setup

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

[MIT License](LICENSE)
