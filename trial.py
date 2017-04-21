import numpy as np
import os
import pickle
from datetime import datetime as dt, timedelta
from read_data import readFile
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def distance(p1, p2):
	return np.sqrt(np.sum(np.square(p1-p2),axis=-1))

def Q(p, x):
	u_p1 = np.mean(p[:,:3],axis=0).reshape((3,1))
	u_p2 = np.mean(x[:,:3],axis=0).reshape((3,1))

	# print(com_p1.shape, com_p2.T.shape)
	u_p1p2 = u_p1.dot(u_p2.T)
	points1 = p[:,:3].reshape((p.shape[0],3,1))
	# print(p[:10])
	# print(points1[:10])
	points2 = x[:,:3].reshape((x.shape[0],3,1))

	p1p2 = np.zeros((points1.shape[0],3,3),dtype=points1.dtype)
	for i in range(points1.shape[0]):
		p1p2[i,:,:] = points1[i].dot(points2[i].T)[:,:]

	# print(p1p2[:10])
	sigma_p1p2 = np.mean(p1p2, axis=0) - u_p1p2
	del p1p2
	del points1
	del points2

	A = sigma_p1p2 - sigma_p1p2.T
	# print(A.shape)

	delta = np.array([[A[1,2]],[A[2,0]],[A[0,1]]],dtype=A.dtype)

	Q_sigma_p1p2 = np.zeros((4,4),dtype=A.dtype)

	Q_sigma_p1p2[0,0] = np.trace(sigma_p1p2)
	Q_sigma_p1p2[1:,0] = delta[:,0]
	Q_sigma_p1p2[0,1:] = delta.T[0,:]
	temp = sigma_p1p2 + sigma_p1p2.T - (np.trace(sigma_p1p2)*np.eye(3))
	Q_sigma_p1p2[1:,1:] = temp[:,:]

	u,s,v = np.linalg.svd(Q_sigma_p1p2)
	# print(u.shape)
	# print(s.shape)
	# print(v.shape)
	# print(np.argmax(s))
	ind = np.argmax(s)

	q_r = v[ind,:]
	# print(np.sum(np.square(q_r)))
	# print(q_r.shape)

	R = np.eye(4,dtype=A.dtype)

	sq_q_r = np.square(q_r)
	R[0,0] = sq_q_r[0] + sq_q_r[1] - sq_q_r[2] - sq_q_r[3]
	R[1,1] = sq_q_r[0] + sq_q_r[2] - sq_q_r[1] - sq_q_r[3]
	R[2,2] = sq_q_r[0] + sq_q_r[3] - sq_q_r[1] - sq_q_r[2]

	R[0,1] = 2*(q_r[1]*q_r[2] + q_r[0]*q_r[3])
	R[1,0] = 2*(q_r[1]*q_r[2] - q_r[0]*q_r[3])

	R[0,2] = 2*(q_r[1]*q_r[3] - q_r[0]*q_r[2])
	R[0,2] = 2*(q_r[1]*q_r[3] + q_r[0]*q_r[2])

	R[1,2] = 2*(q_r[2]*q_r[3] - q_r[0]*q_r[1])
	R[2,1] = 2*(q_r[2]*q_r[3] + q_r[0]*q_r[1])

	q_t = u_p2 - R[:3,:3].dot(u_p1)

	# q = np.hstack((R,q_t))
	# print(R)
	# print(q_t)
	# q_t[0,0] += 0.0000017
	# print(q)
	R[0,3] = q_t[0,0]
	R[1,3] = q_t[1,0]
	R[2,3] = q_t[2,0]
	# rotated_p1 = R.dot(p[:,:3].T)
	# rotated_p1 = p[:,:3].T
	# print(rotated_p1.T[:5])
	# final_p1 = rotated_p1 + q_t*np.ones(rotated_p1.shape)
	final_p1 = R.dot(p.T)
	# print(q_t*np.ones(rotated_p1.shape)[:,:5])
	# print(final_p1.shape)
	final_p1 = final_p1.T

	ms = np.square(distance(x[:,:3],final_p1[:,:3]))
	ms = np.mean(ms, axis = 0)
	return ms, R, q_t

def closest_point(p1, p2):
	p = np.zeros(p1.shape, dtype=p2.dtype)
	inds = np.zeros(p2.shape[0], dtype=np.bool_)
	for i in range(p1.shape[0]):
		temp = p2[np.where(~inds)[0],:3]
		dist = distance(p1[i,:3], temp)
		ind = np.argmin(dist)
		# print(ind.shape)
		# x = ind[0]
		# k = 0
		# while inds[x]:
		# 	k += 1
		# 	x = ind[k]
		# inds[x] = True
		# print(p2[ind,:])
		p[i,:] = p2[ind,:]
	return p

def ICP(p1, p2):
	p0 = p1[:,:]
	R0 = np.eye(3,dtype=np.float32)
	t0 = np.array([[0],[0],[0]],dtype=np.float32)
	m_err = None
	for i in range(1000):
		Y = closest_point(p0, p2)
		ms, R0, t0 = Q(p0, Y)
		p0 = R0.dot(p0.T) #+ t0*np.ones(p0.T.shape)
		p0 = p0.T
		if not (m_err is None):
			diff = m_err - ms
			if diff == 0:
				break
		m_err = ms
		if i%10 == 0:
			print('Mean square error at iteration {}: {}'.format(i, ms))
	# m_err, R0, t0 = Q(p0, p2)
	# p0 = R0.dot(p0.T) # + t0*np.ones(p0.T.shape)
	# p0 = p0.T
	# print('Mean square error at iteration {}: {}'.format(0, m_err))
	return m_err, R0, t0, p0

if __name__ == '__main__':
	dir = 'point_cloud_registration'
	filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
	names = ['pointcloud1','pointcloud2']
	pointcloud1 = readFile('{}/{}'.format(dir, filenames[0]), names[0])
	pointcloud2 = readFile('{}/{}'.format(dir, filenames[1]), names[1])
	
	print(pointcloud1.shape)
	print(pointcloud2.shape)

	# ind = np.where(np.equal(pointcloud1[:,3],pointcloud2[:,3]))[0]

	'''

	Increasing the Lat Long values by a scale of 1000 too increase the mean square error

	'''
	minLat = min(np.min(pointcloud1[:,0]),np.min(pointcloud2[:,0]))
	minLong = min(np.min(pointcloud1[:,1]), np.min(pointcloud2[:,1]))
	pointcloud1 = np.hstack((pointcloud1[:,:3], np.ones((pointcloud1.shape[0],1),dtype=pointcloud1.dtype)))
	pointcloud2 = np.hstack((pointcloud2[:,:3], np.ones((pointcloud2.shape[0],1),dtype=pointcloud2.dtype)))
	# print(pointcloud1.shape)
	# exit()
	scaleLat = 0
	diffLat = np.max(pointcloud1[:,0]) - minLat
	while int(diffLat%10) == 0:
		diffLat *= 10
		scaleLat += 1
	scaleLat += 1
	# print(diffLat, diffLat%10)
	scaleLong = 0
	diffLong = np.max(pointcloud1[:,1]) - minLong
	while int(diffLong%10) == 0:
		diffLong *= 10
		scaleLong += 1
	scaleLong += 1

	cs_Mat = np.eye(4,dtype=pointcloud1.dtype)
	cs_Mat[0,0] *= 10**scaleLat
	cs_Mat[1,1] *= 10**scaleLong
	cs_Mat[0,3] = -minLat*(10**scaleLat)
	cs_Mat[1,3] = -minLong*(10**scaleLong)
	rev_cs_Mat = np.linalg.inv(cs_Mat)
	print('Coordinate system conversion matrix:')
	print(cs_Mat)
	print('Matrix to get back original coordinates:')
	print(rev_cs_Mat)
	# pointcloud1[:,0] -= minLat
	# pointcloud2[:,0] -= minLat
	# pointcloud1[:,1] -= minLong
	# pointcloud2[:,1] -= minLong
	# pointcloud1[:,:2] = pointcloud1[:,:2]*100000
	# pointcloud2[:,:2] = pointcloud2[:,:2]*100000
	# print(pointcloud1[:2,:])

	pc1 = cs_Mat.dot(pointcloud1.T)
	pc1 = pc1.T
	pc2 = cs_Mat.dot(pointcloud2.T)
	pc2 = pc2.T

	# print(pointcloud1[:2,:])

	# pointcloud1 = np.hstack((pointcloud1, np.ones((pointcloud1.shape[0],1), dtype=pointcloud1.dtype)))

	# pointcloud1 = rev_cs_Mat.dot(pointcloud1.T)
	# pointcloud1 = pointcloud1.T
	# print(pointcloud1[:2,:])
	# exit()
	# ind = np.argsort(pointcloud1,axis = 0)
	# print(pointcloud1[ind[0][:10],:])

	# print(np.min(pointcloud1[:,0]), np.max(pointcloud1[:,0]))
	# print(np.min(pointcloud1[:,1]), np.max(pointcloud1[:,1]))
	# print(np.min(pointcloud1[:,2]), np.max(pointcloud1[:,2]))

	# print(np.min(pointcloud2[:,0]), np.max(pointcloud2[:,0]))
	# print(np.min(pointcloud2[:,1]), np.max(pointcloud2[:,1]))
	# print(np.min(pointcloud2[:,2]), np.max(pointcloud2[:,2]))
	# temp = rev_cs_Mat.dot(pointcloud2.T)
	# temp = temp.T

	# with open('p2_1.xyz','w') as file:
	# 	for i in range(temp.shape[0]):
	# 		file.write('{} {} {}\n'.format(temp[i,0],temp[i,1],temp[i,2]))

	# temp = rev_cs_Mat.dot(pointcloud1.T)
	# temp = temp.T

	# with open('p1_1.xyz','w') as file:
	# 	for i in range(temp.shape[0]):
	# 		file.write('{} {} {}\n'.format(temp[i,0],temp[i,1],temp[i,2]))

	# del temp

	ms = None
	print()
	cur = dt.now()
	ms, R, t, final_p1 = ICP(pc1[:10000,:], pc2[:10000,:])
	new_cur = dt.now()
	delt = new_cur - cur
	print('Time Taken: {}'.format(str(delt)))
	# print(ind.shape)

	# d1 = distance(pointcloud1[0,:3], pointcloud1[:,:3])
	# d2 = distance(pointcloud2[0,:3], pointcloud2[:,:3])
	# temp = np.hstack((final_p1, np.ones((final_p1.shape[0],1), dtype = final_p1.dtype)))
	# temp = rev_cs_Mat.dot(temp.T)
	# temp = temp.T
	# with open('p1_fin.xyz','w') as file:
	# 	for i in range(temp.shape[0]):
	# 		file.write('{} {} {}\n'.format(temp[i,0],temp[i,1],temp[i,2]))

	
	print('Mean square error: {}'.format(ms))
	print('Rotation Matrix:')
	print(R[:3,:3])
	print('Translation Matrix:')
	print(t)

	print('\nComplete Transformation Matrix:')
	mat = rev_cs_Mat.dot(R.dot(cs_Mat))
	print(mat)
	# print(pointcloud2[:5])
	# print(final_p1[:5])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X = pointcloud1[:1000,0]
	Y = pointcloud1[:1000,1]
	Z = pointcloud1[:1000,2]

	ax.scatter(X,Y,Z, color='b', marker='o')
	# ax.plot_wireframe(X,Y,Z, color='b')

	X = pointcloud2[:1000,0]
	Y = pointcloud2[:1000,1]
	Z = pointcloud2[:1000,2]

	# ax.plot_wireframe(X,Y,Z, color='r')
	ax.scatter(X,Y,Z, color='r', marker='o')

	ax.set_xlabel('Latitude')
	ax.set_ylabel('Longitude')
	ax.set_zlabel('Altitude')
	
	final_p1 = mat.dot(pointcloud1.T)
	final_p1 = final_p1.T

	X = final_p1[:1000,0]
	Y = final_p1[:1000,1]
	Z = final_p1[:1000,2]

	# ax.plot_wireframe(X,Y,Z, color='g')
	ax.scatter(X,Y,Z, color='g', marker='^')

	plt.show()


	# print(delta.shape)
	
	# print(sigma_p1p2)

	# print(d1)
	# print(d2)

	# d1.sort()
	# d2.sort()

	# ind = np.where(np.equal(d1[:1000],d2[:1000]))
	# print(ind[0].shape)
