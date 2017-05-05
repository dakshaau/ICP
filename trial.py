import numpy as np
import os
import pickle
from datetime import datetime as dt, timedelta
from read_data import readFile
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from icp import icp
from queue import PriorityQueue

class ROTNODE():
	def __init__(self):
		self.a = 0.
		self.b = 0.
		self.c = 0.
		self.w = 0.
		self.ub = 0.
		self.lb = 0.
		self.l = 0

	def __lt__(self, a):
		if self.lb != a.lb:
			return self.lb > a.lb
		else:
			return self.w < a.w
	
	def __str__(self):
		return 'a: {}\nb: {}\nc: {}\nw: {}\nl: {}\nub: {}\nlb: {}'.format(self.a,
			self.b, self.c, self.w, self.l, self.ub, self.lb)


class TRANSNODE():
	def __init__(self):
		self.x = 0.
		self.y = 0.
		self.z = 0.
		self.w = 0.
		self.ub = 0.
		self.lb = 0.

	def __lt__(self, a):
		if self.lb != a.lb:
			return self.lb > a.lb
		else:
			return self.w < a.w

	def __str__(self):
		return 'x: {}\ny: {}\nz: {}\nw: {}\nub: {}\nlb: {}'.format(self.x,
			self.y, self.z, self.w, self.ub, self.lb)


def distance(p1, p2):
	return np.sqrt(np.sum(np.square(p1-p2),axis=-1))

def L2_error(p1, p2):
	return np.sum(np.square(distance(p1,p2)))

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
	# print(p.shape)
	# print(x.shape)
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

	# if np.linalg.det(R) < 0:
	# 	v[2,:] *= -1
	# 	R = v.T.dot(u.T)

	det = np.linalg.det(R)
	temp = np.eye(3)
	temp[2,2] = det

	R = v.T.dot(temp.dot(u.T))

	u_p1 = u_p1.reshape((3,1))
	u_p2 = u_p2.reshape((3,1))
	t = u_p2 - R.dot(u_p1)
	# print(t.shapes)

	X = np.eye(4,dtype=p.dtype)

	X[:3,:3] = R[:,:]
	X[:3,3] = t[:,0]
	R = X[:,:]
	
	return R, R[:3,:3], R[:3,3].reshape((3,1))

def closest_point(p1, tree, p2):
	
	dist, ind = tree.query(p1)
	ind = ind.reshape((ind.shape[0]))
	dist = dist.reshape((dist.shape[0]))
	# print(_[:10])
	p = p2[ind,:]
	del ind
	return p, dist

def ICP(p1, p2, tree, R=np.eye(3), T=np.zeros((3,1))):
	iters = 1000
	p0 = p1[:,:]
	# R0 = None
	X0 = np.eye(4, dtype=np.float32)
	r = R[:,:]
	t = T[:,:]
	# tree = KDTree(p2)
	X0[:3,:3] = r[:,:]
	X0[:3,3] = t[:,0]
	p0 = X0.dot(p0.T)
	p0 = p0.T
	p, _ = closest_point(p0, tree, p2)
	m_err = L2_error(p2[:,:3],p0[:,:3])
	del p
	del _
	# print('Initial Error: {}'.format(m_err))
	X = X0[:,:]
	# print(X)
	for i in range(iters):
		X = X0[:,:]
		Y, _ = closest_point(p0, tree, p2)
		X0, R0, t0 = Q(p0, Y)
		
		# X = X0.dot(X)
		# print('\n{}\n'.format(i))
		# print(X)

		# r = X[:3,:3]
		# t = X[:3,3]

		r = R0.dot(r)
		t = R0.dot(t) + t0
		p0 = X0.dot(p0.T) #+ t0*np.ones(p0.T.shape)
		p0 = p0.T
		ms = L2_error(Y[:,:3],  p0[:,:3])
		# del _
		diff = m_err - ms
		if 0 <= diff < 1e-10:
			# print('Previous error: {}\nCurrent Error: {}\nDifference: {}'.format(m_err, ms, diff))
			m_err = ms
			# print(X)
			break
		m_err = ms
		# if i%10 == 0:
		# 	print('L2 error at iteration {}: {}'.format(i, ms))
			# print(X)
	# X0, r, t = Q(p0, p2)
	# # R0 = createR(q[:4,:],q[4:,:])
	# p0 = X0.dot(p0.T) # + t0*np.ones(p0.T.shape)
	# p0 = p0.T
	# m_err = L2_error(p2[:,:3],p0[:,:3])
	# # print('Mean square error at iteration {}: {}'.format(0, m_err))
	X0[:3,:3] = r[:,:]
	X0[:3,3] = t[:,0]
	# print(X)
	# print(i)
	return m_err, X0, r, t.reshape((3,1)), p0

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


def OuterBnB(p1, p2, initRot, initTrans):
	pi = np.pi
	norm = np.linalg.norm(p1,axis=1)
	global maxrotlvl, maxRotDis, minDis, optR, optT, optError, SSEThresh
	maxrotlvl = 20
	# maxRotDis = np.zeros((maxrotlvl,p1.shape[0]))
	# for i in range(maxrotlvl):
	# 	sigma = initRot.w/(2**i)/2.
	# 	maxAngle = (3**0.5)*sigma

	# 	if maxAngle > pi:
	# 		maxAngle = pi

	# 	maxRotDis[i,:] = 2*np.sin(maxAngle/2)*norm[:]

	# del norm

	tree = KDTree(p2)
	# error, X, R_icp, T_icp, p0 = ICP(p1, p2, tree, R_icp, T_icp)
	# return error, R_icp, T_icp
	

	nodeRot = ROTNODE()
	nodeRotParent = ROTNODE()
	nodeTrans = TRANSNODE()
	queueRot = PriorityQueue()
	p, _ = closest_point(p1,tree,p2)
	del _
	optError = L2_error(p,p1)
	# minDis = distance(p,p1)
	ub = lb = 0.
	print('Error: {} (Init)'.format(optError))
	R_icp = np.eye(3)
	T_icp = np.zeros((3,1))
	cur = dt.now()
	error, X, R_icp, T_icp, p0 = ICP(p1, p2, tree, R_icp, T_icp)
	# return error, R_icp, T_icp
	SSEThresh = 0.001*p1.shape[0]

	if error < optError:
		optError = error
		optR = R_icp[:,:]
		optT = T_icp[:,:]
		new_cur = dt.now()
		delta = new_cur - cur
		print('Error {} (ICP {})'.format(error, str(delta)))
		print('Rotation Mat ICP:')
		print(R_icp)
		print('Translation Mat ICP:')
		print(T_icp)

	queueRot.put(initRot)

	count = 0

	while True:
		if queueRot.empty():
			print('Rotation Queue Empty')
			print('Error: {}, LB: {}'.format(optError,lb))
			break

		nodeRotParent = queueRot.get()
		if optError - nodeRotParent.lb <= SSEThresh:
			print('Error: {}, LB: {}, epsilon: {}'.format(optError, nodeRotParent.lb, SSEThresh))
			break

		if count > 0 and count%300 == 0:
			print('LB={} L={}'.format(nodeRotParent.lb, nodeRotParent.l))
		count += 1

		nodeRot.w = nodeRotParent.w / 2
		nodeRot.l = nodeRotParent.l + 1

		sigma = nodeRot.w / 2
		sigma = (3**0.5)*sigma

		maxRotDis = 2*np.sin(min(sigma/2, np.pi/2))*norm

		for j in range(8):
			
			nodeRot.a = nodeRotParent.a + (j&1) * nodeRot.w
			nodeRot.b = nodeRotParent.b + (j>>1&1) * nodeRot.w
			nodeRot.c = nodeRotParent.c + (j>>2&1) * nodeRot.w

			# print(j, nodeRot)

			v1 = nodeRot.a + nodeRot.w/2
			v2 = nodeRot.b + nodeRot.w/2
			v3 = nodeRot.c + nodeRot.w/2

			if np.linalg.norm([v1,v2,v3]) - (3**0.5)*nodeRot.w/2 > np.pi:
				continue


			t = np.linalg.norm([v1,v2,v3])
			R_ = np.eye(4)
			# p0 = None

			if t > 0:
				v1 /= t
				v2 /= t
				v3 /= t

				ct = np.cos(t)
				ct2 = 1-ct
				st = np.sin(t)
				st2 = 1-st

				tmp121 = v1*v2*ct2; tmp122 = v3*st;
				tmp131 = v1*v3*ct2; tmp132 = v2*st;
				tmp231 = v2*v3*ct2; tmp232 = v1*st;

				R_ = np.eye(4)
				R_[0,0] = ct + v1*v1*ct2;		R_[0,1] = tmp121 - tmp122;		R_[0,2] = tmp131 + tmp132;
				R_[1,0] = tmp121 + tmp122;		R_[1,1] = ct + v2*v2*ct2;		R_[1,2] = tmp231 - tmp232;
				R_[2,0] = tmp131 - tmp132;		R_[2,1] = tmp231 + tmp232;		R_[2,2] = ct + v3*v3*ct2;

				p0 = R_.dot(p0.T)
				p0 = p0.T

			else:
				p0[:,:] = p1[:,:]

			ub, T = InnerBnB(p0, p2, initTrans, R_[:3,:3], None, tree, optError)
			# print(optError)

			if ub < optError:

				optError = ub
				optNodeRot = nodeRot
				optNodeTrans = nodeTrans

				optR[:,:] = R_[:3,:3]

				R_icp[:,:] = R_[:3,:3]
				T_icp[:,:] = T[:,:]

				optT[:,:] = T_icp[:,:]

				print('Error: {}'.format(optError))

				cur = dt.now()
				error, X, R_icp, T_icp, _ = ICP(p1, p2, tree, R_icp, T_icp)
				new_cur = dt.now()
				delta = new_cur-cur

				if error < optError:

					optError = error
					optR[:,:] = R_icp[:,:]
					optT[:,:] = T_icp[:,:]

					print('Error: {} (ICP {})'.format(error, str(delta)))

				queueRotNew = PriorityQueue()
				while not queueRot.empty():
					node = queueRot.get()
					if node.lb < optError:
						queueRotNew.put(node)
					else:
						break

				queueRot = queueRotNew

			# print(maxRotDis.shape)
			# print(nodeRot.l)
			lb, _ = InnerBnB(p0, p2, initTrans, R_[:3,:3], maxRotDis, tree, optError)
			# print(lb)

			if lb >= optError:
				continue

			nodeRot.ub = ub
			nodeRot.lb = lb
			queueRot.put(nodeRot)
			# print(queueRot.qsize())
	# print(count)
	return optError, optR, optT
# p0, p2, nodeTrans, R_, None, tree
def InnerBnB(p0, p2, initTrans, R, maxRotDisL, tree, optError):
	nodeTrans = TRANSNODE()
	queueTrans = PriorityQueue()
	optErrorT = optError

	queueTrans.put(initTrans)
	T = np.zeros((3,1))

	count = 0

	while True:
		if queueTrans.empty():
			break

		nodeTransParent = queueTrans.get()

		if optErrorT - nodeTransParent.lb < SSEThresh:
			break

		count += 1
		nodeTrans.w = nodeTransParent.w/2
		# maxTransDis = (3**0.5)/2.0*nodeTrans.w
		maxTransDis = (3**0.5)*(nodeTrans.w/2)

		for j in range(8):
			nodeTrans.x = nodeTransParent.x + (j&1)*nodeTrans.w
			nodeTrans.y = nodeTransParent.y + (j>>1&1)*nodeTrans.w
			nodeTrans.z = nodeTransParent.z + (j>>2&1)*nodeTrans.w

			
			trans = np.eye(4)
			trans[0,3] = nodeTrans.x + nodeTrans.w/2
			trans[1,3] = nodeTrans.y + nodeTrans.w/2
			trans[2,3] = nodeTrans.z + nodeTrans.w/2
			# trans[:3,:3] = R[:,:]

			p_ = trans.dot(p0.T)
			p_ = p_.T

			p, _ = closest_point(p_, tree, p2)

			minDis = distance(p,p_)

			del p_
			del _

			if not maxRotDisL is None:
				minDis -= maxRotDisL.reshape(minDis.shape)

			ind = np.where(minDis < 0)
			minDis[ind] = 0
			del ind

			ub = np.sum(np.square(minDis))
			# del ind
			
			temp = minDis[:]
			temp = minDis - maxTransDis
			ind = np.where(temp > 0)
			lb = np.sum(np.square(temp[ind]))

			del temp
			del ind

			if ub < optErrorT:
				optErrorT = ub
				T[:,0] = trans[:3,3]
				

			if lb >= optErrorT:
				continue

			nodeTrans.ub = ub
			nodeTrans.lb = lb
			queueTrans.put(nodeTrans)

	# print(count)
	return optErrorT, T

if __name__ == '__main__':
	dir = 'point_cloud_registration'
	filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
	# filenames = ['data_bunny.txt', 'model_bunny.txt']
	# names = ['data','model']
	names = ['pointcloud1','pointcloud2']
	pointcloud1 = readFile('{}/{}'.format(dir, filenames[0]), names[0])
	pointcloud2 = readFile('{}/{}'.format(dir, filenames[1]), names[1])
	
	print(pointcloud1.shape)
	print(pointcloud2.shape)
	if pointcloud1.shape[1] == 4:
		# store available intensities separately
		i1 = pointcloud1[:,3]
		i2 = pointcloud2[:,3]
	else:
		pointcloud1 = np.hstack((pointcloud1, np.ones((pointcloud1.shape[0],1))))
		pointcloud2 = np.hstack((pointcloud2, np.ones((pointcloud2.shape[0],1))))

	minPoints = min(pointcloud1.shape[0],pointcloud2.shape[0])
	# print(np.max(i1))
	# print(i1.shape)

	# ind = np.where(np.equal(pointcloud1[:,3],pointcloud2[:,3]))[0]

	'''

	Increasing the Lat Long values by a scale of 1000 too increase the mean square error

	'''
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
	# pc1 = pointcloud1
	# pc2 = pointcloud2

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

	# exit()
	err = None
	initRot = ROTNODE()
	initTrans = TRANSNODE()

	initRot.a = -np.pi
	initRot.b = -np.pi
	initRot.c = -np.pi
	initRot.w = 2*np.pi
	initRot.l = 0

	initTrans.x = -0.5
	initTrans.y = -0.5
	initTrans.z = -0.5
	initTrans.w = 1

	initTrans.lb = 0.
	initRot.lb = 0.

	print()
	cur = dt.now()
	# minPoints = 10000
	# err, X0, R, T, final_p1 = ICP(pc1[:minPoints,:], pc2[:minPoints,:], KDTree(pc2[:minPoints,:]))
	err, R, T = OuterBnB(pc1[:minPoints,:], pc2[:minPoints,:], initRot, initTrans)
	new_cur = dt.now()
	delt = new_cur - cur
	print('Time Taken: {}'.format(str(delt)))

	print('L2 Error: {}'.format(err))
	print('Rotation Matrix:')
	print(R)
	print('Translation Matrix:')
	print(T)

	X0 = np.eye(4)
	X0[:3,:3] = R[:,:]
	X0[:3,3] = T[:,0]


	# mat = np.eye(4)
	# mat[:3,:3] = R[:,:]
	# mat[:3,3] = T[:,0]

	# print('\nComplete Transformation Matrix:')
	# mat = rev_cs_Mat.dot(X0.dot(cs_Mat))
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
	
	final_p1 = X0.dot(pc1[:minPoints,:].T)
	# final_p1 = rev_cs_Mat.dot(final_p1.T)
	final_p1 = final_p1.T
	# p, _ = closest_point(final_p1[:,:], KDTree(pc2[:minPoints,:]), pc2[:minPoints,:])
	# del _
	# error = L2_error(final_p1, p)
	# print(error)
	# print(np.where(np.equal(final_p1[:,:3], pc1[:,:3])))
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
	ax.set_title('Point Cloud Registration | Visualizing {} points'.format(npoints))

	plt.show()