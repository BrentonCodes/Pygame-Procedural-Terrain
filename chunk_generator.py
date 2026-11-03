import numpy
from numba import njit

import util
import transform


color_ranges = {
	14.1:  (30, 100, 180),
	15.5:  (210, 190, 130),
	20.0:  (60, 130, 50),
	22.5:  (100, 90, 80),
	26.0: (120, 120, 130),
	999.0:(240, 240, 250)
}

# Sorted lookup tables for face coloring, built once at import time.
_color_thresholds = numpy.array(sorted(color_ranges), dtype=numpy.float32)
_color_values = numpy.array(
	[util.rgb_to_packed(color_ranges[t]) for t in sorted(color_ranges)],
	dtype=numpy.uint32
)
_default_color = util.rgb_to_packed((50, 50, 50))

MIN_HEIGHT = 14.0
AMPLITUDE = 40.0
MAX_HEIGHT = MIN_HEIGHT + AMPLITUDE


class Chunk:

	def __init__(self, x, z, s):

		self.position = numpy.array((x*s, 0.0, z*s))
		self.rotation = numpy.array((0.0, 0.0, 0.0))

		self.matrix = transform.create_matrix(self.position, self.rotation)

		self.heightmap = generate_heightmap(x*s, z*s, s)

		self.meshes = {}

		self.height_min = float(self.heightmap.min())
		self.height_max = float(self.heightmap.max())

	def mesh(self, lod):
		"""Return the mesh for a LOD level, generating it on first use."""

		mesh = self.meshes.get(lod)

		if mesh is None:
			mesh = generate_mesh(self.heightmap, lod)
			self.meshes[lod] = mesh

		return mesh

	# Backwards compatible lazy accessors
	@property
	def mesh1(self):
		return self.mesh(1)

	@property
	def mesh2(self):
		return self.mesh(2)

	@property
	def mesh3(self):
		return self.mesh(3)

	@property
	def mesh4(self):
		return self.mesh(4)


# Simplex noise tables, identical to the ones used by the `noise` package.
_PERM = numpy.array([
	151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
	140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
	247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
	57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
	74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
	60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
	65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
	200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
	52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
	207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
	119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
	129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
	218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
	81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
	184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
	222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
	151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
	140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
	247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
	57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
	74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
	60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
	65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
	200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
	52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
	207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
	119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
	129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
	218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
	81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
	184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
	222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
], dtype=numpy.uint8)

_GRAD3 = numpy.array([
	[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
	[1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
	[0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
	[1, 0, -1], [-1, 0, -1], [0, -1, 1], [0, 1, 1],
], dtype=numpy.float32)

_F2 = numpy.float32(0.3660254037844386)
_G2 = numpy.float32(0.21132486540518713)


@njit(cache=True, inline="always")
def _simplex2(x, y):

	s = (x + y) * _F2
	i = numpy.floor(x + s)
	j = numpy.floor(y + s)
	t = (i + j) * _G2

	xx0 = x - (i - t)
	yy0 = y - (j - t)

	i1 = 1 if xx0 > yy0 else 0
	j1 = 1 if xx0 <= yy0 else 0

	xx2 = xx0 + _G2 * 2.0 - 1.0
	yy2 = yy0 + _G2 * 2.0 - 1.0
	xx1 = xx0 - i1 + _G2
	yy1 = yy0 - j1 + _G2

	ii = numpy.int64(i) & 255
	jj = numpy.int64(j) & 255

	g0 = _PERM[ii + _PERM[jj]] % 12
	g1 = _PERM[ii + i1 + _PERM[jj + j1]] % 12
	g2 = _PERM[ii + 1 + _PERM[jj + 1]] % 12

	total = numpy.float32(0.0)

	f = numpy.float32(0.5) - xx0 * xx0 - yy0 * yy0
	if f > 0.0:
		total += f * f * f * f * (_GRAD3[g0, 0] * xx0 + _GRAD3[g0, 1] * yy0)

	f = numpy.float32(0.5) - xx1 * xx1 - yy1 * yy1
	if f > 0.0:
		total += f * f * f * f * (_GRAD3[g1, 0] * xx1 + _GRAD3[g1, 1] * yy1)

	f = numpy.float32(0.5) - xx2 * xx2 - yy2 * yy2
	if f > 0.0:
		total += f * f * f * f * (_GRAD3[g2, 0] * xx2 + _GRAD3[g2, 1] * yy2)

	return total * 70.0


@njit(cache=True)
def _fbm2(x, y, octaves, offset):

	freq = numpy.float32(1.0)
	amp = numpy.float32(1.0)
	total_max = numpy.float32(1.0)

	total = _simplex2(x + offset, y + offset)

	for _ in range(1, octaves):
		freq *= numpy.float32(2.0)
		amp *= numpy.float32(0.5)
		total_max += amp
		total += _simplex2(x * freq + offset, y * freq + offset) * amp

	return total / total_max


@njit(cache=True)
def _generate_heightmap(x, z, s, frequency, amplitude, octaves, offset):

	heightmap = numpy.empty((s + 1, s + 1), dtype=numpy.float32)

	for iz in range(s + 1):

		nz = numpy.float32((z + iz) * frequency)

		for ix in range(s + 1):

			nx = numpy.float32((x + ix) * frequency)

			value = _fbm2(nx, nz, octaves, offset)
			value = (value + numpy.float32(1.0)) * numpy.float32(0.5) * amplitude

			if value < MIN_HEIGHT:
				value = numpy.float32(MIN_HEIGHT)

			heightmap[iz, ix] = value

	return heightmap


def generate_heightmap(x, z, s, frequency=0.01, amplitude=AMPLITUDE, octaves=4, seed=0):

	return _generate_heightmap(
		float(x), float(z), int(s),
		numpy.float32(frequency),
		numpy.float32(amplitude),
		int(octaves),
		numpy.float32(seed)
	)


def generate_mesh(heightmap, lod=1):

	rows, cols = heightmap.shape

	z_samples = numpy.arange(0, rows, lod)
	x_samples = numpy.arange(0, cols, lod)

	if z_samples[-1] != rows - 1:
		z_samples = numpy.append(z_samples, rows - 1)

	if x_samples[-1] != cols - 1:
		x_samples = numpy.append(x_samples, cols - 1)

	sampled = heightmap[numpy.ix_(z_samples, x_samples)]

	sample_rows, sample_cols = sampled.shape

	z_indices, x_indices = numpy.meshgrid(
		z_samples, x_samples, indexing="ij"
	)

	vertices = numpy.empty((sample_rows * sample_cols, 3), dtype=numpy.float32)
	vertices[:, 0] = x_indices.ravel()
	vertices[:, 1] = sampled.ravel()
	vertices[:, 2] = z_indices.ravel()

	grid = numpy.arange(sample_rows * sample_cols, dtype=numpy.int32).reshape(sample_rows, sample_cols)

	a = grid[:-1, :-1].ravel()
	b = grid[:-1, 1:].ravel()
	c = grid[1:, :-1].ravel()
	d = grid[1:, 1:].ravel()

	half = a.shape[0]

	faces = numpy.empty((half * 2, 3), dtype=numpy.int32)
	faces[:half, 0] = a
	faces[:half, 1] = c
	faces[:half, 2] = b
	faces[half:, 0] = b
	faces[half:, 1] = c
	faces[half:, 2] = d

	# Face color from the average height of its three corners
	v0 = vertices[faces[:, 0]]
	v1 = vertices[faces[:, 1]]
	v2 = vertices[faces[:, 2]]

	avg_heights = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / numpy.float32(3.0)

	bucket = numpy.searchsorted(_color_thresholds, avg_heights, side="right")
	in_range = bucket < _color_thresholds.shape[0]

	face_colors = numpy.full(bucket.shape[0], _default_color, dtype=numpy.uint32)
	face_colors[in_range] = _color_values[bucket[in_range]]

	return vertices, faces, face_colors
