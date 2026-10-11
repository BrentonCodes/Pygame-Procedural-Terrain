import pygame
import numpy

import chunk_generator
import transform
import rasterize
import util
import camera

from numba import njit

@njit
def draw_mesh(framebuffer, depthbuffer, points, faces, colrs):

	for i in range(faces.shape[0]):

		f = faces[i]

		v0 = points[f[0]]
		v1 = points[f[1]]
		v2 = points[f[2]]

		if v0[2] > 0 and v1[2] > 0 and v2[2] > 0:

			rasterize.triangle(framebuffer, depthbuffer, colrs[i], v0, v1, v2)

pygame.init()

RESOLUTION = (1280, 720)

sky_color = util.rgb_to_packed((0, 90, 135))

framebuffer = numpy.zeros(RESOLUTION, dtype=numpy.uint32)
depthbuffer = numpy.zeros(RESOLUTION, dtype=numpy.float32)

window = pygame.display.set_mode(RESOLUTION)

pygame.mouse.set_visible(False)
pygame.mouse.set_relative_mode(True)

clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 40)

CHUNK_SIZE = 16
RENDER_DISTANCE = 8

chunks = {}

def get_chunk(chunk_x, chunk_y):
    key = (chunk_x, chunk_y)

    if key not in chunks:
        chunks[key] = chunk_generator.Chunk(
            chunk_x, chunk_y, CHUNK_SIZE
        )

    return chunks[key]

player_camera = camera.Camera()

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

	# GET CHUNKS

	c_chunk_x = int(player_camera.position[0] // CHUNK_SIZE)
	c_chunk_y = int(player_camera.position[2] // CHUNK_SIZE)

	visible_chunks = []

	for x in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
		for y in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
			if x*x + y*y <= RENDER_DISTANCE**2:
				visible_chunks.append(get_chunk(c_chunk_x + x, c_chunk_y + y))

	framebuffer.fill(sky_color)
	depthbuffer.fill(0)

	window.fill(0)

	# RENDER

	view_matrix = transform.create_matrix(player_camera.position, player_camera.rotation, invert=True)

	for chunk in visible_chunks:

		verts = chunk.mesh[0]
		faces = chunk.mesh[1]
		colrs = chunk.mesh[2]

		chunk_matrix = transform.create_matrix(chunk.position, chunk.rotation)

		points = transform.transform(verts, chunk_matrix, view_matrix, player_camera.f)

		draw_mesh(framebuffer, depthbuffer, points, faces, colrs)

	pygame.surfarray.blit_array(window, framebuffer)

	text_surface = font.render(f"FPS: {clock.get_fps():.0f}",False,(255,255,255))

	window.blit(text_surface,(5,5))

	pygame.display.update()