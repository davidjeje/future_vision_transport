# Future Vision Transport — segmentation sémantique Intel XPU

## Installation sous WSL2

```bash
cd future_vision_transport_xpu
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry config virtualenvs.in-project true
rm poetry.lock
poetry lock
poetry install
poetry run python -m ipykernel install --user --name futurevision-xpu-poetry --display-name "FutureVision XPU (Poetry)"
poetry run jupyter lab
```

Le fichier `poetry.lock` fourni est un gabarit : il doit être régénéré sur la machine WSL afin que Poetry résolve les roues Linux XPU exactes.

Déposer les jeux Cityscapes dans `data/` selon les chemins décrits dans le notebook.
