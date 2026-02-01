import os, sys
p = 'code/mask_detector.model'
if not os.path.exists(p):
    print('MISSING')
    sys.exit(0)
if os.path.isdir(p):
    if os.path.exists(os.path.join(p, 'saved_model.pb')):
        print('SAVEDMODEL_DIR')
    else:
        print('DIR_NO_SAVEDMODEL')
    sys.exit(0)
# now file-case: try h5py
try:
    import h5py
except Exception as e:
    print('H5PY_MISSING', str(e))
    sys.exit(0)
try:
    with h5py.File(p,'r'):
        print('HDF5_FILE')
except Exception as e:
    print('NOT_HDF5', str(e))
