# NL-PDDL for ALFWorld: LLM & VLM Baselines

This repository explores Natural Language Planning with PDDL on the ALFWorld benchmark. It includes two baselines:

- LLM baseline that reasons over text observations and goals to generate actions/plans.
- VLM baseline that uses a vision-language model to detect objects in frames and guide action selection.

Both baselines run on ALFWorld environments (TextWorld and/or THOR) with configurable dataset paths.

## Repository layout (key files)
- `run_alfworld_NL.py` — Entry script for the LLM baseline over text observations.
- `run_alfworld_VLM.py` — Entry script for the VLM baseline with object detection; supports resume/failed-only runs.
- `vlm_baseline.py` — Standalone VLM agent loop using ReAct-style prompting (vision + text).
- `llm.py` — LLM helpers:
- `vlm.py` — Vision helpers for Gemini and OpenAI-compatible providers:
- `configs/base_config.yaml` — Main ALFWorld config (env type, dataset paths, training knobs).
- `parser.py`, `plan.py`, `actions.py`, `event.py`, `kb.py` — Core planning, parsing, and knowledge-base logic.
- `alfworld_logic/alfred.pddl`, `alfworld_logic/alfred.twl2` — Domain and grammar files.
- `alfworld/`, `regression_agent/`, `pddl_planner/` — Environment hooks, agents, and planning utilities.

## Requirements
- Python 3.9+ recommended
- Install core dependencies:

```bash
pip install -r requirements.txt
```

Key packages used across scripts:
- ALFWorld stack: `alfworld`, `ai2thor`, `opencv-python`, `numpy`, `Pillow`, `PyYAML`, `tqdm`
- LLM/VLM APIs:
  - OpenAI-compatible: `openai`
  - Google Gemini: `google-genai` (imported as `from google import genai`)
- Visualization and utils: `matplotlib`

If `alfworld` pulls in extra dependencies (PyTorch, gym, etc.), pip will install them; consult ALFWorld’s documentation for platform-specific setup.

## Data setup (ALFWorld)
1. Download ALFWorld JSON datasets (e.g., `json_2.1.1/valid_unseen`).
2. Set an env var pointing to your dataset root:

```bash
export ALFWORLD_DATA=/path/to/alfworld
```

3. In `configs/base_config.yaml`, the defaults use `$ALFWORLD_DATA` for train/eval splits. Some runner scripts also override `data_path` directly — adjust those lines if needed.

## API keys
- OpenAI-compatible providers (for `llm.py`, `vlm.py`, `vlm_baseline.py`):
  - `OPENAI_API_KEY`
  - Optional: `OPENAI_BASE_URL` and `OPENAI_MODEL` if using OpenRouter/Groq/other compatible endpoints
- Google Gemini (for `vlm.py`, `vlm_baseline.py`):
  - `GOOGLE_API_KEY`
- Optional Qwen via DashScope (only in `vlm_baseline.py`):
  - `DASHSCOPE_API_KEY`

Export in your shell before running:
```bash
export OPENAI_API_KEY=...   # or OPENROUTER_API_KEY/GROQ_API_KEY
export GOOGLE_API_KEY=...
export DASHSCOPE_API_KEY=...   # optional
```

## Run: LLM baseline with NLPDDL (text-only)
Main script: `run_alfworld_NL.py`

Notes:
- This script currently sets `env_type = 'AlfredTWEnv'` and overrides `data_path` inside the file. Edit those lines to match your setup or rely solely on `configs/base_config.yaml`.
- The agent uses OpenAI models (e.g., `gpt-4o-mini`) via `llm.py`.

Example:
```bash
python run_alfworld_NL.py
```
This will:
- Walk the dataset and filter solvable tasks.
- For each episode, run the NL regression agent with planning and LLM heuristics.
- Save failed tasks to `failed_tasks.txt`.

## Run: VLM with NLPDDL (vision + text)
Main script: `run_alfworld_VLM.py`

Notes:
- Defaults to `env_type = 'AlfredThorEnv'` and hardcodes `data_path`. Edit these lines to match your setup (or set them in `configs/base_config.yaml`).
- Uses `vlm.detect_object_types()` for object proposals from frames.
- Saves RGB frames with action overlays under `./output_vlm/...`
- Tracks progress in `progress_vlm.json` and failed episodes in `failed_tasks_vlm.txt`.

Basic run:
```bash
python run_alfworld_VLM.py
```

For aligned domain, load `alfworld_domain_original.json`. For misaligned domain, load `alfworld_domain.json`

Resume/rerun only failed episodes from progress file:
```bash
python run_alfworld_VLM.py --use-failed --progress-file progress_vlm.json
```

## Run: VLM with NLPDDL (vision + text)
Other LLM/VLM baselines can be run using `vlm_baseline.py` and `llm_baseline.py`
