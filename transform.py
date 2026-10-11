import numpy

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

def transform(vertices, model_matrix, view_matrix, focal_length):

	v_hom = numpy.pad(vertices, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)

	VM = view_matrix @ model_matrix

	transformed = v_hom @ VM.T

	tx, ty, tz = transformed[:, 0], transformed[:, 1], transformed[:, 2]

	eps = 1e-6
	tz_safe = numpy.where(numpy.abs(tz) < eps, numpy.where(tz < 0, -eps, eps), tz)

	px = (tx / tz_safe) * focal_length
	py = -(ty / tz_safe) * focal_length

	return numpy.stack([px, py, tz], axis=-1)