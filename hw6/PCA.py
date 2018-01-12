import numpy as np
from skimage import io
import sys, os

image_path = sys.argv[1]
img_recon = sys.argv[2]
image = []

#image = io.imread('../hw6_data/Aberdeen/0.jpg')

img_list = os.listdir(image_path)

for i in range(len(img_list)):
  #image_name = image_path + str(i) + '.jpg'
  image_name = image_path + img_list[i]
  img = io.imread(image_name)
  img = np.array(img)
  image.append(img.flatten())

image = np.array(image)

img_mean = image.mean(axis=0)
io.imsave('img_mean.jpg', np.reshape(img_mean, (600,600,3)).astype(np.uint8))

#U, s, V = np.linalg.svd(np.transpose(image-img_mean), full_matrices=False)

#np.save('U.npy', U)
#np.save('s.npy', s)
#np.save('V.npy', V)

U = np.load('U.npy')
s = np.load('s.npy')
V = np.load('V.npy')

ev = np.copy(U)
ev = np.transpose(ev)
for i in range(4):
  image_name = 'ef_'+str(i)+'.jpg'
  ev[i] = ev[i] - np.min(ev[i]) 
  ev[i] = ev[i] / np.max(ev[i]) 
  ev[i] = (ev[i] * 255 * -1) 
  io.imsave(image_name, ev[i].reshape(600, 600, 3).astype(np.uint8))

'''
image_name = 'ef_n_9.jpg'
ev[9] = ev[9] - np.min(ev[9])
ev[9] = ev[9] / np.max(ev[9]) 
ev[9] = (ev[9] * 255 * -1) 
io.imsave(image_name, ev[9].reshape(600, 600, 3).astype(np.uint8))
'''


# reconstruct
ev = np.copy(U)
ev = np.transpose(ev)
k = 4
index = 24
img_re = io.imread(image_path + img_recon)
img_re = np.array(img_re)
img_re = img_re.flatten()
dot_yU = np.dot(ev[:k], img_re-img_mean)
#print(dot_yU)
#print(dot_yU.shape)
#print(U[:k].T.shape)

add_U = np.dot(ev[:k].T, dot_yU) + img_mean
print(add_U.shape)
add_U = add_U - np.min(add_U)
add_U = add_U / np.max(add_U)
add_U = (add_U * 255).astype(np.uint8)

output = 'reconstruction.jpg'
io.imsave(output, add_U.reshape(600,600,3))

#ratio

S = np.copy(s)
print('1: ', (S[0]**2/np.sum(S**2)).astype(np.float64))
print('2: ', (S[1]**2/np.sum(S**2)).astype(np.float64))
print('3: ', (S[2]**2/np.sum(S**2)).astype(np.float64))
print('4: ', (S[3]**2/np.sum(S**2)).astype(np.float64))


