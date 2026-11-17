"""Heightmap terrain renderer: PyOpenGL + GLFW + NumPy (classic VAO/VBO/EBO pipeline).
Terrain is described by a Lua recipe (lupa) whose heavy ops -- fBm noise and hydraulic
erosion -- run in a Rust extension (terrain_core, PyO3).

Controls: WASD move, Space/Ctrl up/down, mouse look, scroll = speed, F = wireframe, Esc = quit.
Usage:  python terrain.py [--recipe terrain.lua] [--size 512] [--shot out.png]
"""
import argparse, ctypes, math, time

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from lupa import LuaRuntime
from PIL import Image

import terrain_core  # Rust (PyO3) module: build with `maturin develop --release -m terrain_core/Cargo.toml`

VS = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
out vec3 vPos; out vec3 vNormal;
void main() { vPos = aPos; vNormal = aNormal; gl_Position = uMVP * vec4(aPos, 1.0); }
"""

FS = """#version 330 core
in vec3 vPos; in vec3 vNormal;
uniform vec3 uEye; uniform float uScale; uniform float uWater;
out vec4 frag;
const vec3 SUN = normalize(vec3(0.6, 0.8, 0.3));
const vec3 SKY = vec3(0.62, 0.75, 0.90);
void main() {
    vec3 n = normalize(vNormal);
    float h = vPos.y / uScale;
    float slope = 1.0 - n.y;
    vec3 c;
    if (h <= uWater + 0.001) c = vec3(0.10, 0.30, 0.50);
    else {
        c = mix(vec3(0.76, 0.70, 0.50), vec3(0.25, 0.50, 0.18), smoothstep(uWater, uWater + 0.04, h));
        c = mix(c, vec3(0.20, 0.38, 0.15), smoothstep(0.45, 0.6, h));
        c = mix(c, vec3(0.45, 0.42, 0.40), smoothstep(0.25, 0.45, slope));   // rock on steep faces
        c = mix(c, vec3(0.95), smoothstep(0.72, 0.80, h) * (1.0 - smoothstep(0.35, 0.55, slope)));
    }
    float diff = max(dot(n, SUN), 0.0);
    vec3 lit = c * (0.30 + 0.75 * diff);
    float fog = 1.0 - exp(-pow(length(vPos - uEye) * 0.0018, 2.0));
    frag = vec4(mix(lit, SKY, fog), 1.0);
}
"""


def run_recipe(path, size):
    """Execute a Lua terrain recipe; it returns {height=..., water=..., height_scale=...}."""
    as_np = lambda b: np.frombuffer(b, np.float32).reshape(size, size).copy()
    opts = lambda t: dict(t.items()) if t else {}

    def image(path):
        img = Image.open(path).convert("F").resize((size, size), Image.BICUBIC)
        h = np.asarray(img, np.float32)
        return (h - h.min()) / max(h.max() - h.min(), 1e-6)

    api = {
        "size": size,
        "fbm": lambda t=None: as_np(terrain_core.fbm(size, **opts(t))),
        "erode": lambda h, t=None: as_np(terrain_core.erode(np.ascontiguousarray(h, np.float32).tobytes(), size, **opts(t))),
        "image": image,
        "mix": lambda a, b, t: a * (1 - t) + b * t,
        "mul": lambda a, b: a * b,
        "pow": lambda h, e: np.clip(h, 0, None) ** e,
    }
    lua = LuaRuntime(register_eval=False, register_builtins=False)
    lua.globals().terrain = lua.table_from(api)
    with open(path) as f:
        result = opts(lua.execute(f.read()))
    h = np.asarray(result["height"], np.float32)
    h = (h - h.min()) / max(h.max() - h.min(), 1e-6)
    return h, float(result.get("water", 0.25)), float(result.get("height_scale", 90))


def build_mesh(h, water, scale):
    """Grid of (x, y, z, nx, ny, nz) vertices + uint32 triangle indices."""
    n = h.shape[0]
    y = np.maximum(h, water) * scale  # flat water surface
    z, x = np.mgrid[0:n, 0:n].astype(np.float32)
    x -= n / 2; z -= n / 2
    dz, dx = np.gradient(y)
    normals = np.dstack([-dx, np.ones_like(y), -dz])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    verts = np.dstack([x, y, z, normals]).reshape(-1, 6).astype(np.float32)

    i = np.arange(n * n, dtype=np.uint32).reshape(n, n)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    idx = np.stack([a, c, b, b, c, d], axis=-1).ravel()
    return verts, idx


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], np.float32)


def look_at(eye, target, up=(0, 1, 0)):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="terrain.lua", help="Lua terrain recipe")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--shot", help="render a few frames, save PNG, exit")
    args = ap.parse_args()

    if not glfw.init():
        raise SystemExit("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    if args.shot:
        glfw.window_hint(glfw.VISIBLE, False)
    win = glfw.create_window(1280, 720, "PyOpenGL terrain", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    t0 = time.perf_counter()
    h, water, height_scale = run_recipe(args.recipe, args.size)
    print(f"recipe {args.recipe}: {time.perf_counter() - t0:.2f}s")
    verts, idx = build_mesh(h, water, height_scale)

    prog = compileProgram(compileShader(VS, GL_VERTEX_SHADER), compileShader(FS, GL_FRAGMENT_SHADER))
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo, ebo = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    for loc, off in ((0, 0), (1, 12)):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(off))

    u_mvp, u_eye = glGetUniformLocation(prog, "uMVP"), glGetUniformLocation(prog, "uEye")
    glUseProgram(prog)
    glUniform1f(glGetUniformLocation(prog, "uScale"), height_scale)
    glUniform1f(glGetUniformLocation(prog, "uWater"), water)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glClearColor(0.62, 0.75, 0.90, 1)

    # camera state
    half = args.size / 2
    eye = np.array([-half * 0.9, height_scale * 1.3, half * 0.9], np.float32)
    yaw, pitch, speed = -45.0, -18.0, 60.0
    wire = False
    state = {"scroll": 0.0, "wire_toggle": False}
    glfw.set_scroll_callback(win, lambda w, dx, dy: state.__setitem__("scroll", state["scroll"] + dy))
    glfw.set_key_callback(win, lambda w, k, s, a, m: state.__setitem__("wire_toggle", True)
                          if k == glfw.KEY_F and a == glfw.PRESS else None)
    if not args.shot:
        glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
    last_mouse = glfw.get_cursor_pos(win)
    last_t, frames, fps_t = time.perf_counter(), 0, time.perf_counter()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = time.perf_counter(); dt = now - last_t; last_t = now
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        mx, my = glfw.get_cursor_pos(win)
        if not args.shot:
            yaw += (mx - last_mouse[0]) * 0.1
            pitch = max(-89.0, min(89.0, pitch - (my - last_mouse[1]) * 0.1))
        last_mouse = (mx, my)
        speed = max(5.0, speed * (1.15 ** state["scroll"])); state["scroll"] = 0.0
        if state["wire_toggle"]:
            wire = not wire; state["wire_toggle"] = False
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wire else GL_FILL)

        fwd = np.array([math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)),
                        math.sin(math.radians(pitch)),
                        math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))], np.float32)
        right = np.cross(fwd, (0, 1, 0)); right /= np.linalg.norm(right)
        for key, vec in ((glfw.KEY_W, fwd), (glfw.KEY_S, -fwd), (glfw.KEY_D, right), (glfw.KEY_A, -right),
                         (glfw.KEY_SPACE, np.array([0, 1, 0])), (glfw.KEY_LEFT_CONTROL, np.array([0, -1, 0]))):
            if glfw.get_key(win, key) == glfw.PRESS:
                eye += vec * speed * dt

        w, hgt = glfw.get_framebuffer_size(win)
        glViewport(0, 0, w, hgt)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        mvp = perspective(60, w / max(hgt, 1), 0.5, 3000) @ look_at(eye, eye + fwd)
        glUniformMatrix4fv(u_mvp, 1, GL_TRUE, mvp)  # row-major numpy -> transpose
        glUniform3fv(u_eye, 1, eye)
        glDrawElements(GL_TRIANGLES, idx.size, GL_UNSIGNED_INT, None)

        frames += 1
        if args.shot and frames == 3:
            data = glReadPixels(0, 0, w, hgt, GL_RGB, GL_UNSIGNED_BYTE)
            Image.frombytes("RGB", (w, hgt), data).transpose(Image.FLIP_TOP_BOTTOM).save(args.shot)
            print(f"saved {args.shot} ({w}x{hgt}, {idx.size // 3} triangles)")
            break
        glfw.swap_buffers(win)
        if now - fps_t > 0.5:
            glfw.set_window_title(win, f"PyOpenGL terrain | {frames / (now - fps_t):.0f} fps | "
                                       f"{idx.size // 3:,} tris | speed {speed:.0f}")
            frames, fps_t = 0, now

    glfw.terminate()


if __name__ == "__main__":
    main()
