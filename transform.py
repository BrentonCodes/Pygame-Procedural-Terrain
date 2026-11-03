import numpy
from numba import njit


def rx_matrix(angle):
	c, s = numpy.cos(angle), numpy.sin(angle)
	return numpy.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=numpy.float32)

def ry_matrix(angle):
	c, s = numpy.cos(angle), numpy.sin(angle)
	return numpy.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=numpy.float32)

def rz_matrix(angle):
	c, s = numpy.cos(angle), numpy.sin(angle)
	return numpy.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=numpy.float32)

def create_matrix(pos, rot, invert=False):

	sign = -1 if invert else 1
	rx, ry, rz = rot
	
	R = rz_matrix(sign * rz) @ rx_matrix(sign * rx) @ ry_matrix(sign * ry)
	T = numpy.eye(4, dtype=numpy.float32)
	T[:3, 3] = sign * pos

	return (R @ T) if invert else (T @ R)

@njit(cache=True, inline="always")
def matmul4(a, b):
	"""4x4 matrix product without depending on an external BLAS."""

	out = numpy.empty((4, 4), dtype=numpy.float32)

	for i in range(4):
		for j in range(4):
			out[i, j] = (
				a[i, 0] * b[0, j] +
				a[i, 1] * b[1, j] +
				a[i, 2] * b[2, j] +
				a[i, 3] * b[3, j]
			)

	return out

@njit(cache=True)
def transform_vertices(vertices, matrix, focal_length, out):
	"""Project vertices with a combined model-view matrix into screen space."""

	n = vertices.shape[0]

	m00 = matrix[0, 0]; m01 = matrix[0, 1]; m02 = matrix[0, 2]; m03 = matrix[0, 3]
	m10 = matrix[1, 0]; m11 = matrix[1, 1]; m12 = matrix[1, 2]; m13 = matrix[1, 3]
	m20 = matrix[2, 0]; m21 = matrix[2, 1]; m22 = matrix[2, 2]; m23 = matrix[2, 3]

	eps = 1e-6

	for i in range(n):

		x = vertices[i, 0]
		y = vertices[i, 1]
		z = vertices[i, 2]

		tx = m00 * x + m01 * y + m02 * z + m03
		ty = m10 * x + m11 * y + m12 * z + m13
		tz = m20 * x + m21 * y + m22 * z + m23

		if tz > -eps and tz < eps:
			tz = -eps if tz < 0.0 else eps

		scale = focal_length / tz

		out[i, 0] = tx * scale
		out[i, 1] = -ty * scale
		out[i, 2] = tz

def transform(vertices, model_matrix, view_matrix, focal_length):
	"""Numpy reference implementation, kept for compatibility."""

	v_hom = numpy.pad(vertices, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)

	VM = view_matrix @ model_matrix

	transformed = v_hom @ VM.T

	tx, ty, tz = transformed[:, 0], transformed[:, 1], transformed[:, 2]

	eps = 1e-6
	tz_safe = numpy.where(numpy.abs(tz) < eps, numpy.where(tz < 0, -eps, eps), tz)

	px = (tx / tz_safe) * focal_length
	py = -(ty / tz_safe) * focal_length

	return numpy.stack([px, py, tz], axis=-1)
