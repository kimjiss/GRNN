from PIL import Image, ImageDraw
import numpy as np
import random


polygon_names = ['triangle', 'rectangle', 'circle']
polygons = {'triangle':[(0, -1), (-1, 1), (1, 1)], 'rectangle': [(-1, -1), (1, -1), (1, 1), (-1, 1)],
			'circle':[(-1, -1), (1, 1)]}
directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

def draw_shape(center, size, shape, color):
	im = Image.new('RGBA', (64, 64), (220, 220, 220))
	draw = ImageDraw.Draw(im)
	if shape=='circle':
		coord = [(center[0] + x * size, center[1] + y * size) for (x, y) in polygons[shape]]
		draw.ellipse(coord, fill=color)
	else:
		coord = [(center[0] + x * size, center[1] + y * size) for (x, y) in polygons[shape]]
		draw.polygon(coord, fill=color)
	return im

def gen_video(datasize):
	file_num = 0
	for iter in range(datasize):
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		shape = polygon_names[random.randint(0, 2)]
		size = random.randint(3, 9)
		dir = directions[random.randint(0, 7)]
		center = [32, 32]
		for i in range(4):
			img = draw_shape(center, size, shape, color)
			img.save('./data/toy/%d.bmp'%file_num)
			file_num += 1
			center[0] += 6 * dir[0]
			center[1] += 6 * dir[1]
		if iter % 5000 ==0:
			print '%d videos generated'%iter


gen_video(30000)
