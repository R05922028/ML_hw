from PIL import Image
import sys

imagename = sys.argv[1]
im = Image.open(imagename)
pix_ori = im.load()
pix = list(im.getdata())
pix = [(int(pixel[0]/2), int(pixel[1]/2), int(pixel[2]/2)) for pixel in pix]
im.putdata(pix)

fout = open("Q2.txt","w")
for i in range(im.size[0]):
	for j in range(im.size[1]):
		r,g,b = im.getpixel((i,j))
		fout.write(str(r))
		fout.write(' ')
		fout.write(str(g))
		fout.write(' ')
		fout.write(str(b))
		fout.write('\n')
im.save("Q2.png")
