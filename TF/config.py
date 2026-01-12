import numpy as np
import torch
import os


class GlobalConfig:
    bev_h = 128
    bev_w = 256
    front_h = bev_h
    front_w = bev_w
    hz = 4 #1 detik ada berapa sample yang direcord
    bias_basic = 15
    bearing_bias = [-bias_basic, bias_basic, 2*bias_basic+5, bias_basic, -bias_basic+10, -bias_basic] #dalam derajat #bias untuk 0 ke 60, 60 ke 120, 120 ke 180, -180 ke -120, -120 ke -60, -60 ke 0
    rp1_close = 4 #jarak minimum untuk ganti rp1 (dalam meter)   

    #for training
    gpu_id = '0'
    model = 'transfuser' #late_fusion, geometric_fusion, transfuser
    logdir = 'log/'+model
    init_stop_counter = 30
    batch_size = 10
    lr = 1e-4 # learning rate #pakai AdamW
    weight_decay = 1e-3
    n_fmap_r18 = [64, 64, 128, 256, 512]
    n_fmap_r34 = [64, 64, 128, 256, 512] #sama dengan resnet18

	# Data
    seq_len = 1 # jumlah input seq
    pred_len = 4 # future waypoints predicted
    n_wp = pred_len #waypoints

    """
    #buat geometric fusion
    n_scale = 4 
    vert_anchors = 4#8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors
    n_embd = 512
    """

    #buat transfuser
    n_views = 1 # no. of camera views
    vert_anchors = 4#8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    logdir = logdir+"_seq"+str(seq_len) #update direktori name
    # root_dir = '/media/aisl/data/oskar/ros-whill-robot2/main/dataset/dataset'
    # root_dir = '/home/aisl/OSKAR/WHILL/ros-whill-robot2/main/dataset/dataset'
    root_dir = os.path.dirname(os.getcwd())+'/dataset/dataset'
    train_dir = root_dir+'/train_routes'
    val_dir = root_dir+'/val_routes'
    test_dir = root_dir+'/test_routes'
    train_conditions = ['noon', 'evening', 'night'] #
    val_conditions = ['noon', 'evening', 'night'] #pokoknya kebalikannya train
    test_conditions = ['noon0', 'evening0', 'night0',
                        'noon1', 'evening1', 'night1',
                        'noon2', 'evening2', 'night2'] 

    # Controller
    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15 # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15 # buffer size

    max_throttle = 1.0 # upper limit on throttle signal value in dataset
    wheel_radius = 0.15#radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1 #minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    #buat preprocessing data
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    cover_area_lr = 16 #kiri - kanan
    cover_area_f = [1.25, 17.25] #posisi camera -> area interest max
    
    #lidar setting, cek HDL-32E dan VLP32C LiDAR sensor datasheet
    lidar_sensor = "hdl32e" #vlp32c hdl32e
    if lidar_sensor == "hdl32e":
        v_fov = [-30.67, 10.67] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 100#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
        v_res_div = 55
    else: #"vlp32c"
        v_fov = [-25, 15] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
        v_res_div = 55
    max_intensity = 100.0
    # v_fov_down = -1*np.radians(2)
    # v_fov_up = np.radians(24.9)
    # n_laser = 32
    # lidar_rps = 10 #rotasi per detik --> 600 rpm / 60 detik
    h_fov = 360
    # v_fov = [-25, 15] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
    v_fov_total = -v_fov[0] + v_fov[1]

    v_res = v_fov_total/v_res_div         #n_laser #front_h  # 1.33 #vertical resolution
    h_res = h_fov/(front_w*2)              #0.35 #horizontal resolution
    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)
    # y_fudge = 5

    #untuk front_dep dan bev_dep
    #100 untuk HDL32E, 200 untuk VLP32C
    # dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
    dep_min = cover_area_f[0]

    #other, buat join_img dll
    fps = 20
    rgb_res_ori = [720, 1280] #HxW
    scale_w = rgb_res_ori[1]/front_w
    scale_h = rgb_res_ori[0]/front_h

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
