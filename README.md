# Example of Dual-Arm Diff IK

Example of dual-arm diff IK in Drake, with slight modification for collision avoidance.

This formulation is not "perfect" yet, as it can still have odd behavior in certain locations.

Note that this only focused on *kinematic* control, focused on a dual-arm system.
For model-based torque control of a single-arm system, see:
<https://github.com/EricCousineau-TRI/drake_torque_control_study>


## Setup

Be sure you have `poetry`: <https://python-poetry.org>

```sh
poetry install
```

## Running

```sh
# For each terminal.
poetry shell

# (Separate terminal) Visualizer
python -m pydrake.visualization.meldis -w

# (Main terminal) Running
python ./scripts/main_example.py
```

![Demo](/vid.gif)
