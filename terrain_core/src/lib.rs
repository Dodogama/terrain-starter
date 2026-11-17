//! CPU-heavy terrain generation exposed to Python via PyO3.
//! Heightmaps cross the boundary as raw little-endian f32 bytes (numpy `tobytes()` / `frombuffer`),
//! so there's no per-element conversion and no numpy crate dependency.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn to_f32(data: &[u8], size: usize) -> PyResult<Vec<f32>> {
    if data.len() != size * size * 4 {
        return Err(PyValueError::new_err("buffer length != size*size*4"));
    }
    Ok(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn to_bytes<'py>(py: Python<'py>, h: &[f32]) -> Bound<'py, PyBytes> {
    let out: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    PyBytes::new(py, &out)
}

/// Tiny deterministic RNG (xorshift64*), avoids pulling in `rand`.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x27d4eb2d) ^ (y as u32).wrapping_mul(0x165667b1) ^ seed.wrapping_mul(0x9e3779b9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    (h & 0xffffff) as f32 / 0xffffff as f32
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = hash(x0, y0, seed) + (hash(x0 + 1, y0, seed) - hash(x0, y0, seed)) * sx;
    let b = hash(x0, y0 + 1, seed) + (hash(x0 + 1, y0 + 1, seed) - hash(x0, y0 + 1, seed)) * sx;
    a + (b - a) * sy
}

/// Fractal value noise normalized to 0..1. `ridged` folds each octave (1-|2n-1|)^2 for sharp crests.
#[pyfunction]
#[pyo3(signature = (size, octaves=8, scale=4.0, persistence=0.5, lacunarity=2.0, seed=0, ridged=false))]
fn fbm<'py>(py: Python<'py>, size: usize, octaves: u32, scale: f32, persistence: f32,
            lacunarity: f32, seed: u32, ridged: bool) -> Bound<'py, PyBytes> {
    let mut h = vec![0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            let (mut amp, mut freq, mut sum) = (1.0, scale / size as f32, 0.0);
            for o in 0..octaves {
                let mut n = value_noise(x as f32 * freq, y as f32 * freq, seed.wrapping_add(o * 101));
                if ridged {
                    n = (1.0 - (2.0 * n - 1.0).abs()).powi(2);
                }
                sum += n * amp;
                amp *= persistence;
                freq *= lacunarity;
            }
            h[y * size + x] = sum;
        }
    }
    let (lo, hi) = h.iter().fold((f32::MAX, f32::MIN), |(l, u), &v| (l.min(v), u.max(v)));
    h.iter_mut().for_each(|v| *v = (*v - lo) / (hi - lo).max(1e-6));
    to_bytes(py, &h)
}

/// Height + gradient at a fractional position via bilinear interpolation.
fn sample(h: &[f32], n: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let (ix, iy) = (x as usize, y as usize);
    let (u, v) = (x - ix as f32, y - iy as f32);
    let i = iy * n + ix;
    let (a, b, c, d) = (h[i], h[i + 1], h[i + n], h[i + n + 1]);
    let gx = (b - a) * (1.0 - v) + (d - c) * v;
    let gy = (c - a) * (1.0 - u) + (d - b) * u;
    let height = a * (1.0 - u) * (1.0 - v) + b * u * (1.0 - v) + c * (1.0 - u) * v + d * u * v;
    (height, gx, gy)
}

/// Droplet hydraulic erosion: each drop runs downhill, picks up sediment where it speeds up
/// and deposits where it slows / climbs. `strength` scales heights during simulation so slopes
/// on a 0..1 map behave like real terrain; tune it from the Lua recipe.
#[pyfunction]
#[pyo3(signature = (data, size, drops=100_000, seed=1, strength=64.0, radius=3))]
fn erode<'py>(py: Python<'py>, data: &[u8], size: usize, drops: u32, seed: u64,
              strength: f32, radius: i32) -> PyResult<Bound<'py, PyBytes>> {
    let mut h = to_f32(data, size)?;
    h.iter_mut().for_each(|v| *v *= strength);
    let n = size;
    let (inertia, capacity_k, min_cap, erode_k, deposit_k, evap, gravity, lifetime) =
        (0.05f32, 4.0f32, 0.01f32, 0.3f32, 0.3f32, 0.01f32, 4.0f32, 30);

    // erosion brush: cone weights within `radius`, normalized
    let mut brush = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let w = radius as f32 - ((dx * dx + dy * dy) as f32).sqrt();
            if w > 0.0 { brush.push((dx, dy, w)); }
        }
    }
    let total: f32 = brush.iter().map(|b| b.2).sum();
    brush.iter_mut().for_each(|b| b.2 /= total);

    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    py.detach(|| {
        for _ in 0..drops {
            let (mut x, mut y) = (rng.next_f32() * (n - 2) as f32, rng.next_f32() * (n - 2) as f32);
            let (mut dx, mut dy, mut speed, mut water, mut sediment) = (0f32, 0f32, 1f32, 1f32, 0f32);
            for _ in 0..lifetime {
                let (ix, iy) = (x as usize, y as usize);
                let (u, v) = (x - ix as f32, y - iy as f32);
                let (height, gx, gy) = sample(&h, n, x, y);
                dx = dx * inertia - gx * (1.0 - inertia);
                dy = dy * inertia - gy * (1.0 - inertia);
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 { break; }
                dx /= len; dy /= len;
                x += dx; y += dy;
                if x < 0.0 || y < 0.0 || x >= (n - 2) as f32 || y >= (n - 2) as f32 { break; }

                let dh = sample(&h, n, x, y).0 - height;
                let cap = (-dh * speed * water * capacity_k).max(min_cap);
                if sediment > cap || dh > 0.0 {
                    // deposit on the 4 cells of the old position, bilinear weights
                    let amt = if dh > 0.0 { dh.min(sediment) } else { (sediment - cap) * deposit_k };
                    sediment -= amt;
                    let i = iy * n + ix;
                    h[i] += amt * (1.0 - u) * (1.0 - v);
                    h[i + 1] += amt * u * (1.0 - v);
                    h[i + n] += amt * (1.0 - u) * v;
                    h[i + n + 1] += amt * u * v;
                } else {
                    let amt = ((cap - sediment) * erode_k).min(-dh);
                    for &(bx, by, w) in &brush {
                        let (cx, cy) = (ix as i32 + bx, iy as i32 + by);
                        if cx < 0 || cy < 0 || cx >= n as i32 || cy >= n as i32 { continue; }
                        let c = &mut h[cy as usize * n + cx as usize];
                        let take = (amt * w).min(*c);
                        *c -= take;
                        sediment += take;
                    }
                }
                speed = (speed * speed + dh * gravity).max(0.0).sqrt();
                water *= 1.0 - evap;
            }
        }
    });
    h.iter_mut().for_each(|v| *v /= strength);
    Ok(to_bytes(py, &h))
}

#[pymodule]
fn terrain_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fbm, m)?)?;
    m.add_function(wrap_pyfunction!(erode, m)?)?;
    Ok(())
}
