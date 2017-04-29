import numpy as np
import os
import pickle
from datetime import datetime as dt, timedelta
from read_data_py27 import readFile
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
# from icp import icp
from pcl import _pcl, registration as reg

def distance(p1, p2):
	return np.sqrt(np.sum(np.square(p1-p2),axis=-1))

def createR(q_r, q_t):
	R = np.eye(4, dtype=q_r.dtype)
	sq_q_r = np.square(q_r)
	R[0,0] = sq_q_r[0,0] + sq_q_r[1,0] - sq_q_r[2,0] - sq_q_r[3,0]
	R[1,1] = sq_q_r[0,0] + sq_q_r[2,0] - sq_q_r[1,0] - sq_q_r[3,0]
	R[2,2] = sq_q_r[0,0] + sq_q_r[3,0] - sq_q_r[1,0] - sq_q_r[2,0]
	del sq_q_r

	R[0,1] = 2*(q_r[1,0]*q_r[2,0] + q_r[0,0]*q_r[3,0])
	R[1,0] = 2*(q_r[1,0]*q_r[2,0] - q_r[0,0]*q_r[3,0])

	R[0,2] = 2*(q_r[1,0]*q_r[3,0] - q_r[0,0]*q_r[2,0])
	R[0,2] = 2*(q_r[1,0]*q_r[3,0] + q_r[0,0]*q_r[2,0])

	R[1,2] = 2*(q_r[2,0]*q_r[3,0] - q_r[0,0]*q_r[1,0])
	R[2,1] = 2*(q_r[2,0]*q_r[3,0] + q_r[0,0]*q_r[1,0])

	R[:3,3] = q_t[:,0]

	return R

def Q(p, x):
	u_p1 = np.mean(p[:,:3],axis=0)#.reshape((3,1))
	u_p2 = np.mean(x[:,:3],axis=0)#.reshape((3,1))

	# print(com_p1.shape, com_p2.T.shape)
	u_p1p2 = u_p1.dot(u_p2.T)
	points1 = p[:,:3]#.reshape((p.shape[0],3,1))
	# print(p[:10])
	# print(points1[:10])
	points2 = x[:,:3]#.reshape((x.shape[0],3,1))
	p1 = points1 - u_p1
	p2 = points2 - u_p2

	p1p2 = p1.T.dot(p2)
	u,s,v = np.linalg.svd(p1p2)

	R = v.T.dot(u.T)

	if np.linalg.det(R) < 0:
		v[2,:] *= -1
		R = v.T.dot(u.T)

	u_p1 = u_p1.reshape((3,1))
	u_p2 = u_p2.reshape((3,1))
	t = u_p2 - R.dot(u_p1)
	# print(t.shapes)

	X = np.eye(4,dtype=p.dtype)

	X[:3,:3] = R[:,:]
	X[:3,3] = t[:,0]
	R = X[:,:]
	
	return R, R[:3,:3], R[:3,3] 

def closest_point(p1, tree, p2):
	
	ind = tree.query(p1, return_distance = False)
	ind = ind.reshape((ind.shape[0]))
	# print(_[:10])
	p = p2[ind,:]
	del ind
	return p

# def ICP(p1, p2):
# 	iters = 1000
# 	p0 = p1[:,:]
# 	R0 = None
# 	X0 = np.eye(4, dtype=np.float32)
# 	tree = KDTree(p2)
# 	p0 = X0.dot(p0.T)
# 	p0 = p0.T
# 	t = None
# 	m_err = np.mean(distance(p2,p0),axis=0)
# 	print('Initial Error: {}'.format(m_err))
# 	for i in range(iters):
# 		Y = closest_point(p0, tree, p2)
# 		X0, R0, t = Q(p0, Y)
# 		p0 = X0.dot(p0.T) #+ t0*np.ones(p0.T.shape)
# 		p0 = p0.T
# 		ms = distance(Y[:,:3], p0[:,:3])
# 		ms = np.mean(ms,axis=0)
# 		diff = m_err - ms
# 		if 0 < diff < 1e-10:
# 			print('Previous error: {}\nCurrent Error: {}\nDifference: {}'.format(m_err, ms, diff))
# 			break
# 		m_err = ms
# 		if i%10 == 0:
# 			print('Mean square error at iteration {}: {}'.format(i, ms))
# 	# X0, R0, t = Q(p0, p2)
# 	# # R0 = createR(q[:4,:],q[4:,:])
# 	# p0 = X0.dot(p0.T) # + t0*np.ones(p0.T.shape)
# 	# p0 = p0.T
# 	# m_err = np.mean(distance(p2[:,:3],p0[:,:3]),axis=0)
# 	# print('Mean square error at iteration {}: {}'.format(0, m_err))
# 	return m_err, X0, R0, t, p0

def get_degree_to_meter(ref, points):
	pi = np.pi
	arcsin = lambda x: np.arcsin(x)
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	ER = 6371000.

	p = points*(pi/180.)
	ab = ref - p
	PD = 2*ER * arcsin(np.sqrt( sin(ab[:,0]/2.)**2 + cos(0)*cos(p[:,0])*(sin(ab[:,1]/2.))**2) )
	return PD

def get_meter_to_degree(ref, points, slat, slng):
	ER = 6371000.
	pi = np.pi
	sin = lambda x: np.sin(x)
	cos = lambda x: np.cos(x)
	arcsin = lambda x: np.arcsin(x)
	lat = (ref[0] - points[:,0]/ER)*(180./pi)
	# numer = np.square()

	lng = (ref[1] - 2*arcsin(np.sqrt((np.square(sin(points[:,1]/(2*ER))) + np.square(sin(ref[0]/2.)))/cos(ref[0]))))*(180./pi)
	sign = 1
	if np.min(lat) < 0:
		sign *= -1
	if sign != slat:
		lat = lat * -1
	sign = 1
	if np.min(lng) < 0:
		sign *= -1
	if sign != slng:
		lng = lng * -1
	return np.hstack((lat.reshape((lat.shape[0],1)), lng.reshape((lng.shape[0],1))))

if __name__ == '__main__':
	dir = 'point_cloud_registration'
	filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
	names = ['pointcloud1_py27','pointcloud2_py27']
	pointcloud1 = readFile('%s/%s'%(dir, filenames[0]), names[0])
	pointcloud2 = readFile('%s/%s'%(dir, filenames[1]), names[1])
	
	print(pointcloud1.shape)
	print(pointcloud2.shape)
	i1 = pointcloud1[:,3]
	i2 = pointcloud2[:,3]
	# print(np.max(i1))
	# print(i1.shape)

	# ind = np.where(np.equal(pointcloud1[:,3],pointcloud2[:,3]))[0]

	'''

	Increasing the Lat Long values by a scale of 1000 too increase the mean square error

	'''
	# minLat = min(np.min(pointcloud1[:,0]),np.min(pointcloud2[:,0]))
	# minLong = min(np.min(pointcloud1[:,1]), np.min(pointcloud2[:,1]))
	# pointcloud1 = np.hstack((pointcloud1[:,:3], np.ones((pointcloud1.shape[0],1),dtype=pointcloud1.dtype)))
	# pointcloud2 = np.hstack((pointcloud2[:,:3], np.ones((pointcloud2.shape[0],1),dtype=pointcloud2.dtype)))
	# # print(pointcloud1.shape)
	# # exit()
	# scaleLat = 0
	# diffLat = np.max(pointcloud1[:,0]) - minLat
	# while int(diffLat%10) == 0:
	# 	diffLat *= 10
	# 	scaleLat += 1
	# scaleLat += 1
	# # print(diffLat, diffLat%10)
	# scaleLong = 0
	# diffLong = np.max(pointcloud1[:,1]) - minLong
	# while int(diffLong%10) == 0:
	# 	diffLong *= 10
	# 	scaleLong += 1
	# scaleLong += 1

	signLat = 1
	if np.min(pointcloud1[:,0]) < 0:
		signLat *= -1
	signLong = 1
	if np.min(pointcloud1[:,1]) < 0:
		signLong *= -1

	pc1 = np.zeros(pointcloud1.shape, dtype=pointcloud1.dtype)
	pc2 = np.zeros(pointcloud2.shape, dtype=pointcloud2.dtype)

	pc1[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud1[:,0].reshape((pc1.shape[0],1)), np.zeros((pc1.shape[0],1), dtype=np.float32))))[:]
	pc1[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((pc1.shape[0],1), dtype=np.float32), pointcloud1[:,1].reshape((pc1.shape[0],1)))))[:]
	pc1[:,2] = pointcloud1[:,2]
	pc1[:,3] = 1.

	pc2[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud2[:,0].reshape((pc2.shape[0],1)), np.zeros((pc2.shape[0],1), dtype=np.float32))))[:]
	pc2[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((pc2.shape[0],1), dtype=np.float32), pointcloud2[:,1].reshape((pc2.shape[0],1)))))[:]
	pc2[:,2] = pointcloud2[:,2]
	pc2[:,3] = 1.

	minX = min(np.min(pc1[:,0]), np.min(pc2[:,0]))
	minY = min(np.min(pc1[:,1]), np.min(pc2[:,1]))

	# dist = np.zeros((2,2))
	# print(pointcloud1[:2,:2])

	# dist[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud1[:2,0].reshape((dist.shape[0],1)), np.zeros((dist.shape[0],1), dtype=np.float32))))[:]
	# dist[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((dist.shape[0],1), dtype=np.float32), pointcloud1[:2,1].reshape((dist.shape[0],1)))))[:]
	# print(dist)

	# p = get_meter_to_degree(np.array([0.,0.]), dist, signLat, signLong)
	# print(p)
	# print(minX, minY)
	# dist[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((p[:2,0].reshape((dist.shape[0],1)), np.zeros((dist.shape[0],1), dtype=np.float32))))[:]
	# dist[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((dist.shape[0],1), dtype=np.float32), p[:2,1].reshape((dist.shape[0],1)))))[:]
	# print(dist)

	# exit()
	cs_Mat = np.eye(4,dtype=pointcloud1.dtype)
	# cs_Mat[0,0] *= 10**scaleLat
	# cs_Mat[1,1] *= 10**scaleLong
	cs_Mat[0,3] = -minX
	cs_Mat[1,3] = -minY
	rev_cs_Mat = np.linalg.inv(cs_Mat)
	# print('Coordinate system conversion matrix:')
	# print(cs_Mat)
	# print('Matrix to get back original coordinates:')
	# print(rev_cs_Mat)
	# # pointcloud1[:,0] -= minLat
	# # pointcloud2[:,0] -= minLat
	# # pointcloud1[:,1] -= minLong
	# # pointcloud2[:,1] -= minLong
	# # pointcloud1[:,:2] = pointcloud1[:,:2]*100000
	# # pointcloud2[:,:2] = pointcloud2[:,:2]*100000
	# # print(pointcloud1[:2,:])
	# print(pointcloud2[:2,:2])
	# dist = np.zeros((2,2))
	# dist[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud2[:2,0].reshape((2,1)), np.zeros((2,1)))))
	# dist[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((2,1)), pointcloud2[:2,1].reshape((2,1)))))
	# print(dist)

	# p = get_meter_to_degree(np.array([0.,0.]), dist, signLat, signLong)
	# print(p)

	pc1 = cs_Mat.dot(pc1.T)
	pc1 = pc1.T
	pc2 = cs_Mat.dot(pc2.T)
	pc2 = pc2.T

	# temp = rev_cs_Mat.dot(pc2.T)
	# temp = temp.T

	# with open('p2_1.pts','w') as file:
	# 	for i in range(temp.shape[0]):
	# 		file.write('{},{},{},{}\n'.format(temp[i,0],temp[i,1],temp[i,2],i2[i]))

	# temp = rev_cs_Mat.dot(pc1.T)
	# temp = temp.T

	# with open('p1_1.pts','w') as file:
	# 	for i in range(temp.shape[0]):
	# 		file.write('{},{},{},{}\n'.format(temp[i,0],temp[i,1],temp[i,2],i1[i]))

	# del temp
	# exit()

	# T, dist = icp(pc1[:,:3], pc2[:,:3])

	# print(T, dist)

	p1 = _pcl.PointCloud(np.float32(pc1[:,:3]))
	p2 = _pcl.PointCloud(np.float32(pc2[:,:3]))
	cur = dt.now()
	c, T, p, err = reg.icp(p1, p2, max_iter = 1000)
	new_cur = dt.now()
	delt = new_cur - cur
	print('Time Taken: %s'%(str(delt)))

	final_p1 = p.to_array()
	print('Error %f'%(err))
	print('Transformation Matrix:')
	print(T)

	# # exit()
	# ms = None
	# print()
	
	# ms, X, R, t, final_p1 = ICP(pc1[:,:], pc2[:,:])

	# print('Mean square error: {}'.format(ms))
	# print('Rotation Matrix:')
	# print(R)
	# print('Translation Matrix:')
	# print(t)

	# print('\nComplete Transformation Matrix:')
	# mat = rev_cs_Mat.dot(X.dot(cs_Mat))
	# print(mat)
	# print(pointcloud2[:5])
	# print(final_p1[:5])
	# pc1 = rev_cs_Mat.dot(pc1.T)
	# pc1 = pc1.T
	# pc1[:,:2] = get_meter_to_degree(np.array([0.,0.]), pc1[:,:2], signLat, signLong)[:,:]
	# pc2 = rev_cs_Mat.dot(pc2.T)
	# pc2 = pc2.T
	# pc2[:,:2] = get_meter_to_degree(np.array([0.,0.]), pc2[:,:2], signLat, signLong)[:,:]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	npoints = 10000

	X = pc1[:npoints,0]
	Y = pc1[:npoints,1]
	Z = pc1[:npoints,2]

	# ax.scatter(X,Y,Z, color='b', marker='o', label='Point Cloud 1')
	# ax.plot_wireframe(X,Y,Z, color='b')

	X = pc2[:npoints,0]
	Y = pc2[:npoints,1]
	Z = pc2[:npoints,2]

	# ax.plot_wireframe(X,Y,Z, color='r')
	ax.scatter(X,Y,Z, color='r', marker='o',label='Point Cloud 2')

	ax.set_xlabel('Latitude')
	ax.set_ylabel('Longitude')
	ax.set_zlabel('Altitude')
	
	# final_p1 = mat.dot(pointcloud1.T)
	# final_p1 = rev_cs_Mat.dot(final_p1.T)
	# final_p1 = final_p1.T
	# final_p1[:,:2] = get_meter_to_degree(np.array([0.,0.]), final_p1[:,:2], signLat, signLong)[:,:]
	# with open('p1_fin.pts','w') as file:
	# 	for i in range(final_p1.shape[0]):
	# 		file.write('{},{},{},{}\n'.format(final_p1[i,0],final_p1[i,1],final_p1[i,2],i1[i]))

	X = final_p1[:npoints,0]
	Y = final_p1[:npoints,1]
	Z = final_p1[:npoints,2]

	# ax.plot_wireframe(X,Y,Z, color='g')
	ax.scatter(X,Y,Z, color='g', marker='o', label='Registered Point Cloud')
	ax.legend()
	ax.set_title('Point Cloud Registration | Visualizing %d points'%(npoints))

	plt.show()