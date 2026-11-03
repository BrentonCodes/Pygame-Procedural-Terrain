import math

import numpy
from numba import njit

import transform


@njit(cache=True, inline="always")
def edge(x0, y0, x1, y1, x2, y2):
	return (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)


@njit(cache=True, inline="always")
def _ceil_offset(t):
	# smallest integer >= t, clamped so the conversion can never overflow
	if t <= 0.0:
		return 0

	if t >= 1073741824.0:
		return 1 << 30

	return int(math.ceil(t))


@njit(cache=True, inline="always")
def _floor_offset(t):
	# largest integer <= t; negative means the row is empty
	if t < 0.0:
		return -1

	if t >= 1073741824.0:
		return 1 << 30

	return int(math.floor(t))


@njit(cache=True, inline="always")
def _triangle(fb, zb, color, p0, p1, p2, w, h):

	x0, y0, z0 = p0
	x1, y1, z1 = p1
	x2, y2, z2 = p2

	cx = w * 0.5
	cy = h * 0.5

	x0 += cx; y0 += cy
	x1 += cx; y1 += cy
	x2 += cx; y2 += cy

	area = edge(x0, y0, x1, y1, x2, y2)

	# back-face / degenerate triangle
	if area >= 0.0:
		return

	minx = max(0, int(min(x0, x1, x2)))
	maxx = min(w - 1, int(max(x0, x1, x2)))
	miny = max(0, int(min(y0, y1, y2)))
	maxy = min(h - 1, int(max(y0, y1, y2)))

	if maxx < minx or maxy < miny:
		return

	inv_area = 1.0 / area

	# Barycentric gradients along x (A) and y (B)
	A0 = (y2 - y1) * inv_area; B0 = (x1 - x2) * inv_area
	A1 = (y0 - y2) * inv_area; B1 = (x2 - x0) * inv_area
	A2 = (y1 - y0) * inv_area; B2 = (x0 - x1) * inv_area

	# Reciprocal x-slopes so each scanline only needs multiplications
	rA0 = 1.0 / A0 if A0 != 0.0 else 0.0
	rA1 = 1.0 / A1 if A1 != 0.0 else 0.0
	rA2 = 1.0 / A2 if A2 != 0.0 else 0.0

	# Barycentric weights at (minx, miny)
	w0_row = ((minx - x1) * (y2 - y1) - (miny - y1) * (x2 - x1)) * inv_area
	w1_row = ((minx - x2) * (y0 - y2) - (miny - y2) * (x0 - x2)) * inv_area
	w2_row = ((minx - x0) * (y1 - y0) - (miny - y0) * (x1 - x0)) * inv_area

	iz0 = 1.0 / z0
	iz1 = 1.0 / z1
	iz2 = 1.0 / z2

	# Interpolated 1/z changes by a constant amount for every pixel step in x
	iz_step = A0 * iz0 + A1 * iz1 + A2 * iz2

	for y in range(miny, maxy + 1):

		xa = minx
		xb = maxx
		empty = False

		# Clip the scanline to the triangle by solving w >= 0 for x
		if A0 > 0.0:
			xi = minx + _ceil_offset(-w0_row * rA0)
			if xi > xa:
				xa = xi
		elif A0 < 0.0:
			xi = minx + _floor_offset(-w0_row * rA0)
			if xi < xb:
				xb = xi
		elif w0_row < 0.0:
			empty = True

		if not empty:
			if A1 > 0.0:
				xi = minx + _ceil_offset(-w1_row * rA1)
				if xi > xa:
					xa = xi
			elif A1 < 0.0:
				xi = minx + _floor_offset(-w1_row * rA1)
				if xi < xb:
					xb = xi
			elif w1_row < 0.0:
				empty = True

		if not empty:
			if A2 > 0.0:
				xi = minx + _ceil_offset(-w2_row * rA2)
				if xi > xa:
					xa = xi
			elif A2 < 0.0:
				xi = minx + _floor_offset(-w2_row * rA2)
				if xi < xb:
					xb = xi
			elif w2_row < 0.0:
				empty = True

		if not empty and xa <= xb:

			iz = w0_row * iz0 + w1_row * iz1 + w2_row * iz2 + (xa - minx) * iz_step

			for x in range(xa, xb + 1):

				if iz > zb[x, y]:
					fb[x, y] = color
					zb[x, y] = iz

				iz += iz_step

		w0_row += B0
		w1_row += B1
		w2_row += B2


@njit(cache=True)
def triangle(fb, zb, color, p0, p1, p2):
	_triangle(fb, zb, color, p0, p1, p2, fb.shape[0], fb.shape[1])


@njit(cache=True)
def draw_mesh(fb, zb, vertices, faces, colors, model_matrix, view_matrix, focal_length, points):

	matrix = transform.matmul4(view_matrix, model_matrix)

	transform.transform_vertices(vertices, matrix, focal_length, points)

	w = fb.shape[0]
	h = fb.shape[1]

	for i in range(faces.shape[0]):

		f = faces[i]

		v0 = points[f[0]]
		v1 = points[f[1]]
		v2 = points[f[2]]

		if v0[2] > 0.0 and v1[2] > 0.0 and v2[2] > 0.0:
			_triangle(fb, zb, colors[i], v0, v1, v2, w, h)
