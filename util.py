def rgb_to_packed(color):

	r, g, b = color

	return (int(r) << 16) | (int(g) << 8) | int(b)