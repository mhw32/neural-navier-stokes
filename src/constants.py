import os
SRC_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SRC_DIR, 'data')
CHORIN_FD_DATA_FILE = os.path.join(DATA_DIR, 'chorin_fd', 'data_semi_implicit.npz')
DIRECT_FD_DATA_FILE = os.path.join(DATA_DIR, 'direct_fd', 'data.npz')
