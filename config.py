class Config(object):
    data_path = r'D:\3dgan\data'
    g_learning_rate = 0.0025
    d_learning_rate = 0.00001
    batch_size = 5
    max_epoch = 200
    noise_dim = 200
    d_train = 1
    g_train = 5
    save_epoch = 20
    save_img = r'D:\3dgan\log_1\imgs'
    save_model = r'D:\3dgan\log_1\model'
    netg_path = None
    netd_path = None

