# Point Cloud Registration

**Language:** Python 3.6

### Decription:

Register two 3D point clouds of the same area using Iterative Closest Point, or ICP.

Tried to implement [Go-ICP](http://iitlab.bit.edu.cn/mcislab/~yangjiaolong/go-icp/), but the results were not good, meybe due to incorrect implementation. The implementation is still in the code, but is not called upon. The code calls ICP by default.

The code in `read_data.py` is used to load text data into numpy arrays and create pickle dumps  on the first run, to save time when called later.

Error function used is **L2 Error**.


### To Run:

```Batchfile
python trial.py
```