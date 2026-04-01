#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$REPO_ROOT/docs"

# Add local Quarto install to PATH if present
if [ -d "$HOME/.local/quarto/bin" ]; then
  export PATH="$HOME/.local/quarto/bin:$PATH"
fi

echo "=== Syncing tutorial notebook ==="
cp "$REPO_ROOT/examples/blockworld_tutorial.ipynb" "$DOCS/tutorial.ipynb"

echo "=== Syncing tutorial images ==="
mkdir -p "$DOCS/images"
cp "$REPO_ROOT/examples/images/"* "$DOCS/images/"

python3 -c "
import json, re
nb_path = '$DOCS/tutorial.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

# Inject YAML front-matter as a raw cell if not already present
front_matter = {
    'cell_type': 'raw',
    'metadata': {'raw_mimetype': 'text/markdown'},
    'source': ['---\n', 'title: \"Tutorial: Blockworld with Misalignment\"\n',
               'execute:\n', '  enabled: false\n', '---\n']
}
if not (nb['cells'] and nb['cells'][0].get('cell_type') == 'raw'
        and any('---' in s for s in nb['cells'][0].get('source', []))):
    nb['cells'].insert(0, front_matter)
    print('  Injected YAML front-matter cell')

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        # Replace --- horizontal rules so Quarto does not parse them as YAML
        cell['source'] = [
            line.replace('---', '***') if line.strip() == '---' else line
            for line in cell['source']
        ]
        # Rewrite attachment:filename refs to images/filename
        cell['source'] = [
            re.sub(r'\(attachment:([^)]+)\)', r'(images/\1)', line)
            for line in cell['source']
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print('  Notebook patched (front-matter, rules, image refs)')
"

echo "=== Generating tree viewer demo HTML ==="
RESULT_FILE="${1:-$DOCS/_assets/demo_result.txt}"
python -m pddl_planner.plan_viewer "$RESULT_FILE" \
    -o "$DOCS/_assets/plan_viewer_demo.html" --no-open

echo "=== Rendering Quarto site ==="
cd "$DOCS"
quarto render .

echo "=== Done ==="
echo "Preview locally with:  quarto preview $DOCS"
