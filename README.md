# NetworkIntrusionDetector

Project: Reinforcement-learning based Network Intrusion Detection (modularized)

## Structure
See the repository layout in this project. Notebooks and scripts are provided to preprocess data, build features, define a Gym environment, train a DQN agent (Stable-Baselines3), and evaluate the agent.

## Quick start
1. Create and activate conda env from `environment.yml` or install from `requirements.txt`.
2. Place raw dataset CSV files into `data/raw/`.
3. Run `python src/data/make_dataset.py` to create `data/interim/` and `data/processed/`.
4. Run `python src/models/train_agent.py` to train a DQN agent (or open the notebooks to run cells interactively).
5. Evaluate with `python src/models/evaluate_agent.py`.

## Notes
- The notebooks are split into modular steps for clarity.
- Models and scalers are saved under `models/`.
- This is a template: adapt reward shaping and environment logic in `src/env/intrusion_env.py` to match your dataset and task.
