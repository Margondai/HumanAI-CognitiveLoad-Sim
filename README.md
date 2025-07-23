# HumanAI-CognitiveLoad-Sim

A digital twin simulation framework for evaluating emotion-aware AI in human-AI teaming for defense operations, including adaptive load management and comparison with static AI systems.

This framework models 100 agent archetypes (novices to veterans) over 30,000 simulation steps, integrating:

- **Physiological inputs:** HRV, GSR, and simulated EEG ratios  
- **Markov-based emotional state transitions:** Calm, Stressed, Overloaded  
- **Support Vector Machine (SVM) classification** for state detection  
- **Adaptive AI logic** to dynamically adjust workload and penalties  
- **Comparison with static, emotion-agnostic AI**

## Repository Structure

- `simulation_framework.py` – Core simulation engine  
- *(Optional)* `run_simulation.py` – Example runner (to quickly test the sim)  
- `data/` – Folder for storing generated or synthetic datasets  
- `analysis/` – Scripts for visualization and statistical analysis  
- `docs/` – Documentation and references (e.g., MODSIM 2025 paper)  

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/HumanAI-CognitiveLoad-Sim.git
cd HumanAI-CognitiveLoad-Sim

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # (We'll generate this later if needed)

Quick Start
To run a basic simulation with 100 agents over 30,000 steps:

bash
Copy code
python simulation_framework.py
This will output a sample dataset showing each agent’s Calm vs. Overloaded time (for testing).

For more detailed analysis (visualizations, statistics), see analysis/ (coming soon).

License
This project is licensed under the MIT License – see the LICENSE file for details.

Citation
If you use this framework in your research, please cite:

Margondai, A., Von Ahlefeldt, C., Willox, S., & Mouloua, M. (2025). Emotion-Aware Cognitive Load Management in Human-AI Teaming for Defense Operations. MODSIM World 2025.


