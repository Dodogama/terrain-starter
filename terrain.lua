-- Terrain recipe. Heavy ops (fbm, erode) run in Rust (terrain_core); Lua just composes them.
-- Available: terrain.size, terrain.fbm{...}, terrain.ridged{...}, terrain.erode(h, {...}),
--            terrain.image(path), terrain.mix(a, b, t), terrain.mul(a, b), terrain.pow(h, e)

local base   = terrain.fbm{octaves = 8, scale = 3, seed = 7}
local ridges = terrain.fbm{octaves = 6, scale = 5, seed = 11, ridged = true}

-- ridges only where the base is already high -> mountain ranges inland, smooth coasts
local h = terrain.mix(base, terrain.mul(ridges, base), 0.5)
h = terrain.pow(h, 1.4)
h = terrain.erode(h, {drops = 250000, seed = 3, strength = 64})

return { height = h, water = 0.22, height_scale = 90 }
