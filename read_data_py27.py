import numpy as np
import os
import pickle
from datetime import datetime as dt, timedelta

def readFile(file, name):
	if not os.path.exists(name):
		data = np.loadtxt(file, delimiter=' ')
		pickle.dump(data, open(name,'wb'))
		return data
	else:
		data = pickle.load(open(name,'rb'))
		return data

if __name__ == '__main__':
	cur = dt.now()
	dir = 'point_cloud_registration'
	filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
	names = ['pointcloud1_py27','pointcloud2_py27']
	data = readFile('%s/%s'%(dir,filenames[1]),names[1])
	new_cur = dt.now()
	delta = new_cur - cur
	print('Time Taken: %f'%(delta.total_seconds()))
	print(data.shape)