import numpy
import noise
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

class Chunk:

	def __init__(self, x, z, s):

		self.position = numpy.array((x*s, 0.0, z*s))
		self.rotation = numpy.array((0.0, 0.0, 0.0))

		self.matrix = transform.create_matrix(self.position, self.rotation)

		self.heightmap = generate_heightmap(x*s, z*s, s)

		self.mesh1 = generate_mesh(self.heightmap, lod=1)
		self.mesh2 = generate_mesh(self.heightmap, lod=2)
		self.mesh3 = generate_mesh(self.heightmap, lod=3)
		self.mesh4 = generate_mesh(self.heightmap, lod=4)

def generate_heightmap(x, z, s, frequency=0.01, amplitude=40.0, octaves=4, seed=0):

	map_data = []

	for iz in range(s+1):

		row = []

		for ix in range(s+1):

			nx = (x + ix) * frequency
			nz = (z + iz) * frequency

			value = noise.snoise2(nx, nz, octaves=octaves, base=seed, persistence=0.5, lacunarity=2.0)
			value = (value + 1) / 2
			value = value * amplitude

			value = max(14,value)

			row.append(value)

		map_data.append(row)

	return numpy.asarray(map_data, dtype=numpy.float64)

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

	vertices = numpy.column_stack((
		x_indices.ravel(),
		sampled.ravel(),
		z_indices.ravel()
	))

	grid = numpy.arange(sample_rows * sample_cols).reshape(sample_rows, sample_cols)

	a = grid[:-1, :-1].ravel()
	b = grid[:-1, 1:].ravel()
	c = grid[1:, :-1].ravel()
	d = grid[1:, 1:].ravel()

	faces = numpy.vstack((
		numpy.column_stack((a, c, b)),
		numpy.column_stack((b, c, d))
	))

	v0 = vertices[faces[:, 0]]
	v1 = vertices[faces[:, 1]]
	v2 = vertices[faces[:, 2]]

	normals = numpy.cross(v1 - v0, v2 - v0)

	lengths = numpy.linalg.norm(normals, axis=1, keepdims=True)
	face_normals = normals / numpy.maximum(lengths, 1e-12)

	# Calculate colors

	avg_heights = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
	face_colors = numpy.empty(len(faces), dtype=numpy.uint32)

	for i, h in enumerate(avg_heights):
		rgb = (50, 50, 50)

		for threshold, color in color_ranges.items():
			if h < threshold:
				rgb = color
				break

		face_colors[i] = util.rgb_to_packed(rgb)

	return vertices, faces, face_colors