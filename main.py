import pygame
import numpy
from numba import njit

import chunk_generator
import transform
import rasterize
import util
import camera


RESOLUTION = (1280, 720)

CHUNK_SIZE = 64
RENDER_DISTANCE = 6

# Squared LOD switch distances
LOD1_DISTANCE_SQ = 128.0 * 128.0
LOD2_DISTANCE_SQ = 256.0 * 256.0
LOD3_DISTANCE_SQ = 512.0 * 512.0

# World height range used for conservative frustum culling of whole chunks.
CHUNK_HEIGHT_MIN = chunk_generator.MIN_HEIGHT
CHUNK_HEIGHT_MAX = chunk_generator.MAX_HEIGHT + 0.01

# Every chunk offset inside the render circle, precomputed once.
_offsets = numpy.array(
	[
		(x, z)
		for x in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1)
		for z in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1)
		if x * x + z * z <= RENDER_DISTANCE * RENDER_DISTANCE
	],
	dtype=numpy.int64
)
_offsets_x = numpy.ascontiguousarray(_offsets[:, 0])
_offsets_z = numpy.ascontiguousarray(_offsets[:, 1])


@njit(cache=True)
def frustum_cull(offset_x, offset_z, base_x, base_z, chunk_size, y_min, y_max, view, tan_x, tan_y):
	"""Return a mask of chunks whose bounding boxes intersect the view frustum."""

	n = offset_x.shape[0]
	visible = numpy.zeros(n, dtype=numpy.bool_)

	m00 = view[0, 0]; m01 = view[0, 1]; m02 = view[0, 2]; m03 = view[0, 3]
	m10 = view[1, 0]; m11 = view[1, 1]; m12 = view[1, 2]; m13 = view[1, 3]
	m20 = view[2, 0]; m21 = view[2, 1]; m22 = view[2, 2]; m23 = view[2, 3]

	inf = 1e30

	for i in range(n):

		x0 = (base_x + offset_x[i]) * chunk_size
		z0 = (base_z + offset_z[i]) * chunk_size

		min_x = inf; max_x = -inf
		min_y = inf; max_y = -inf
		min_z = inf; max_z = -inf

		for corner in range(8):

			px = x0 + (chunk_size if corner & 1 else 0.0)
			py = y_max if corner & 2 else y_min
			pz = z0 + (chunk_size if corner & 4 else 0.0)

			vx = m00 * px + m01 * py + m02 * pz + m03
			vy = m10 * px + m11 * py + m12 * pz + m13
			vz = m20 * px + m21 * py + m22 * pz + m23

			if vx < min_x: min_x = vx
			if vx > max_x: max_x = vx
			if vy < min_y: min_y = vy
			if vy > max_y: max_y = vy
			if vz < min_z: min_z = vz
			if vz > max_z: max_z = vz

		if max_z <= 0.0:
			continue

		# Conservative test of the view-frustum side planes
		limit_x = max_z * tan_x
		limit_y = max_z * tan_y

		if min_x <= limit_x and max_x >= -limit_x and min_y <= limit_y and max_y >= -limit_y:
			visible[i] = True

	return visible


@njit(cache=True)
def frustum_cull_boxes(world_x, world_z, y_min, y_max, chunk_size, view, tan_x, tan_y):
	"""Frustum test using each chunk's own vertical bounds."""

	n = world_x.shape[0]
	visible = numpy.zeros(n, dtype=numpy.bool_)

	m00 = view[0, 0]; m01 = view[0, 1]; m02 = view[0, 2]; m03 = view[0, 3]
	m10 = view[1, 0]; m11 = view[1, 1]; m12 = view[1, 2]; m13 = view[1, 3]
	m20 = view[2, 0]; m21 = view[2, 1]; m22 = view[2, 2]; m23 = view[2, 3]

	inf = 1e30

	for i in range(n):

		x0 = world_x[i]
		z0 = world_z[i]

		min_x = inf; max_x = -inf
		min_y = inf; max_y = -inf
		max_z = -inf

		for corner in range(8):

			px = x0 + (chunk_size if corner & 1 else 0.0)
			py = y_max[i] if corner & 2 else y_min[i]
			pz = z0 + (chunk_size if corner & 4 else 0.0)

			vx = m00 * px + m01 * py + m02 * pz + m03
			vy = m10 * px + m11 * py + m12 * pz + m13
			vz = m20 * px + m21 * py + m22 * pz + m23

			if vx < min_x: min_x = vx
			if vx > max_x: max_x = vx
			if vy < min_y: min_y = vy
			if vy > max_y: max_y = vy
			if vz > max_z: max_z = vz

		if max_z <= 0.0:
			continue

		limit_x = max_z * tan_x
		limit_y = max_z * tan_y

		if min_x <= limit_x and max_x >= -limit_x and min_y <= limit_y and max_y >= -limit_y:
			visible[i] = True

	return visible


def main():

	pygame.init()

	sky_color = util.rgb_to_packed((0, 90, 135))

	framebuffer = numpy.zeros(RESOLUTION, dtype=numpy.uint32)
	depthbuffer = numpy.zeros(RESOLUTION, dtype=numpy.float32)

	window = pygame.display.set_mode(RESOLUTION)

	pygame.mouse.set_visible(False)
	pygame.mouse.set_relative_mode(True)

	clock = pygame.time.Clock()

	font = pygame.font.SysFont(None, 40)

	half_width = RESOLUTION[0] * 0.5
	half_height = RESOLUTION[1] * 0.5

	chunks = {}

	# Chunks further than this are dropped from memory occasionally so that
	# long flights don't grow the chunk store forever.
	EVICT_RADIUS = RENDER_DISTANCE + 2
	evict_timer = 0.0

	player_camera = camera.Camera()

	tan_x = half_width / player_camera.f
	tan_y = half_height / player_camera.f

	hud_surface = None
	hud_timer = 0.0

	# Scratch space for projected chunk vertices, reused every chunk
	transformed_vertices = numpy.empty(((CHUNK_SIZE + 1) * (CHUNK_SIZE + 1), 3), dtype=numpy.float32)

	running = True

	while running:

		dt = clock.tick(0) / 1000

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				running = False

			if event.type == pygame.KEYDOWN:

				if event.key == pygame.K_ESCAPE:

					running = False

		player_camera.update(dt)

		px, py, pz = player_camera.position

		c_chunk_x = int(px // CHUNK_SIZE)
		c_chunk_z = int(pz // CHUNK_SIZE)

		# Keep the chunk store bounded
		evict_timer -= dt

		if evict_timer <= 0.0:

			evict_timer = 2.0

			stale = [
				key for key in chunks
				if abs(key[0] - c_chunk_x) > EVICT_RADIUS or abs(key[1] - c_chunk_z) > EVICT_RADIUS
			]

			for key in stale:
				del chunks[key]

		view_matrix = transform.create_matrix(player_camera.position, player_camera.rotation, invert=True)

		# FRUSTUM CULLING

		visible = frustum_cull(
			_offsets_x, _offsets_z,
			c_chunk_x, c_chunk_z,
			CHUNK_SIZE,
			CHUNK_HEIGHT_MIN, CHUNK_HEIGHT_MAX,
			view_matrix, tan_x, tan_y
		)

		chunk_indices = numpy.nonzero(visible)[0]

		n_chunks = chunk_indices.shape[0]

		# Second, tighter pass using the actual height range of chunks we
		# already know about. Unknown chunks keep the global height range.
		if n_chunks:

			world_x = (c_chunk_x + _offsets_x[chunk_indices]) * float(CHUNK_SIZE)
			world_z = (c_chunk_z + _offsets_z[chunk_indices]) * float(CHUNK_SIZE)

			y_min = numpy.empty(n_chunks, dtype=numpy.float32)
			y_max = numpy.empty(n_chunks, dtype=numpy.float32)

			for j in range(n_chunks):

				i = chunk_indices[j]

				chunk = chunks.get((
					c_chunk_x + int(_offsets_x[i]),
					c_chunk_z + int(_offsets_z[i])
				))

				if chunk is None:
					y_min[j] = CHUNK_HEIGHT_MIN
					y_max[j] = CHUNK_HEIGHT_MAX
				else:
					y_min[j] = chunk.height_min
					y_max[j] = chunk.height_max

			visible = frustum_cull_boxes(
				world_x, world_z, y_min, y_max,
				CHUNK_SIZE, view_matrix, tan_x, tan_y
			)

			chunk_indices = chunk_indices[visible]
			world_x = world_x[visible]
			world_z = world_z[visible]

			n_chunks = chunk_indices.shape[0]

		# Sort near chunks first so early depth testing rejects more fragments.
		if n_chunks:

			dx = world_x - px
			dy = -py
			dz = world_z - pz

			dist_sq = dx * dx + dy * dy + dz * dz

			order = numpy.argsort(dist_sq)

			chunk_indices = chunk_indices[order]
			dist_sq = dist_sq[order]

		else:

			dist_sq = numpy.zeros(0, dtype=numpy.float64)

		framebuffer.fill(sky_color)
		depthbuffer.fill(0)

		# RENDER

		for j in range(chunk_indices.shape[0]):

			i = chunk_indices[j]

			chunk_x = c_chunk_x + int(_offsets_x[i])
			chunk_z = c_chunk_z + int(_offsets_z[i])

			key = (chunk_x, chunk_z)

			chunk = chunks.get(key)

			if chunk is None:
				chunk = chunk_generator.Chunk(chunk_x, chunk_z, CHUNK_SIZE)
				chunks[key] = chunk

			distance_sq = dist_sq[j]

			if distance_sq <= LOD1_DISTANCE_SQ:
				lod = 1
			elif distance_sq <= LOD2_DISTANCE_SQ:
				lod = 2
			elif distance_sq <= LOD3_DISTANCE_SQ:
				lod = 3
			else:
				lod = 4

			mesh = chunk.mesh(lod)

			rasterize.draw_mesh(
				framebuffer, depthbuffer,
				mesh[0], mesh[1], mesh[2],
				chunk.matrix, view_matrix,
				player_camera.f,
				transformed_vertices
			)

		pygame.surfarray.blit_array(window, framebuffer)

		# HUD (re-rendered a few times per second instead of every frame)
		hud_timer -= dt

		if hud_surface is None or hud_timer <= 0.0:

			hud_surface = font.render(f"FPS: {clock.get_fps():.0f}", False, (255, 255, 255))
			hud_timer = 0.25

		window.blit(hud_surface, (5, 5))

		pygame.display.update()


if __name__ == "__main__":
	main()
