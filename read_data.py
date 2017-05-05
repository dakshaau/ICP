import numpy as np
import os
import pickle
from datetime import datetime as dt, timedelta

def readFile(file, name):
	if not os.path.exists(name):
		data = np.loadtxt(file, delimiter=' ')#, skiprows=1)
		pickle.dump(data, open(name,'wb'))
		return data
	else:
		data = pickle.load(open(name,'rb'))
		return data

if __name__ == '__main__':
	cur = dt.now()
	dir = 'point_cloud_registration'
	# filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
	filenames = ['data_bunny.txt', 'model_bunny.txt']
	names = ['data','model']
	# names = ['pointcloud1','pointcloud2']
	data = readFile('{}/{}'.format(dir,filenames[1]),names[1])
	new_cur = dt.now()
	delta = new_cur - cur
	print('Time Taken: {}'.format(delta.total_seconds()))
	print(data.shape)