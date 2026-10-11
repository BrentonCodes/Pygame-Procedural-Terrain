import numpy
import noise
import util

color_ranges = {
	4.1:  (30, 100, 180),
	5.5:  (210, 190, 130),
	8.0:  (60, 130, 50),
	10.5:  (100, 90, 80),
	12.0: (120, 120, 130),
	999.0:(240, 240, 250)
}

class Chunk:

	def __init__(self, x, z, s):

		self.position = numpy.array((x*s, 0.0, z*s))
		self.rotation = numpy.array((0.0, 0.0, 0.0))

		self.heightmap = generate_heightmap(x*s, z*s, s)
		self.mesh = generate_mesh(self.heightmap)

def generate_heightmap(x, z, s, frequency=0.03, amplitude=20.0, octaves=4, seed=0):

	map_data = []

	for iz in range(s+1):

		row = []

		for ix in range(s+1):

			nx = (x + ix) * frequency
			nz = (z + iz) * frequency

			value = noise.snoise2(nx, nz, octaves=octaves, base=seed)
			value = (value + 1) / 2
			value = value * amplitude

			row.append(value)

		map_data.append(row)

	return numpy.asarray(map_data, dtype=numpy.float64)

def generate_mesh(heightmap):

	rows, cols = heightmap.shape

	z_indices, x_indices = numpy.indices((rows, cols))

	vertices = numpy.column_stack((x_indices.ravel(), heightmap.ravel(), z_indices.ravel()))

	grid = numpy.arange(rows * cols).reshape(rows, cols)

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

	# CALCULATE COLORS
	
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