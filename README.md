# terrain-starter

Heightmap terrain with **PyOpenGL + GLFW**: a classic VAO/VBO/EBO pipeline with slope/height shading and fog.
The terrain is described by a **Lua** recipe (`terrain.lua`, run through `lupa`). The recipe's heavy ops,
fBm noise and droplet hydraulic erosion, run in a **Rust** extension (`terrain_core/`, PyO3 + maturin).

```
Lua recipe ──calls──▶ Python API ──bytes──▶ Rust (fbm, erode) ──▶ NumPy mesh ──▶ PyOpenGL
```

## Setup
```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
source ~/.cargo/env   # rustup toolchain
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m terrain_core/Cargo.toml
```

## Run
```sh
.venv/bin/python terrain.py                    # uses terrain.lua
.venv/bin/python terrain.py --recipe my.lua --size 1024
.venv/bin/python terrain.py --shot out.png     # headless screenshot
.venv/bin/python test_core.py                  # Rust module check
```
Controls: WASD and mouse to fly, Space/Ctrl to move up/down, scroll to change speed, F for wireframe, Esc to quit.
Edit `terrain.lua` to change the terrain without recompiling. `terrain.image("map.png")` loads a grayscale heightmap.
