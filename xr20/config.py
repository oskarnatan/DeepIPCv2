import numpy as np
import torch
import os


class GlobalConfig:
    bev_h = 128
    bev_w = 256
    front_h = 64
    front_w = 512
    # w = 256
    hz = 4 #1 detik ada berapa sample yang direcord
    bias_basic = 15
    bearing_bias = [-bias_basic, bias_basic, 2*bias_basic+5, bias_basic, -bias_basic+10, -bias_basic] #dalam derajat #bias untuk 0 ke 60, 60 ke 120, 120 ke 180, -180 ke -120, -120 ke -60, -60 ke 0
    rp1_close = 4 #jarak minimum untuk ganti rp1 (dalam meter)   

    #for training
    gpu_id = '0'
    inputs = 'segdep' #segdep seg dep
    # perspectives = 'bevfro' #bevfro bev fro
    logdir = 'log/xr20_'+inputs#+'_'+perspectives
    init_stop_counter = 30
    batch_size = 10
    lr = 1e-4 # learning rate #pakai AdamW
    weight_decay = 1e-3
    #parameter untuk MGN
    MGN = True
    loss_weights = [1, 1, 1] #wp, mlp st, mlp th
    lw_alpha = 1.5
    bottleneck = 64 # #cek dengan check_arch.py
    n_fmap = [48, 96, 192, 384]

	# Data
    seq_len = 1 # jumlah input seq
    pred_len = 3 # future waypoints predicted
    n_wp = pred_len #waypoints
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
    #control weights untuk PID dan MLP dari tuningan MGN
    #urutan steering, throttle
    #baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    #LWS: lw_wp lw_str lw_thr saat convergence
    lws = [1, 1, 1]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2])] #str, thrt
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1]] #str, thrt, brk

    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15 # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15 # buffer size

    n_cmd = 3 #jumlah command yang ada: 0 lurus, 1 kiri, 2 kanan
    max_throttle = 1.0 # upper limit on throttle signal value in dataset
    wheel_radius = 0.15#radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1 #minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    #buat preprocessing data
    polarseg_weight_path = os.path.join(os.getcwd(), "polarseg/SemKITTI_PolarSeg.pt")
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    cover_area_lr = 16 #kiri - kanan
    cover_area_up = [-1.5, 6.5] #bawah -> atas
    cover_area_f = [1.25, 17.25] #posisi camera -> area interest max
    SEG_CLASSES = { #lihat di file semantic-kitti.yaml
        'colors'        :[[0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80],
                        [255, 0, 0], [30, 30, 255], [200, 40, 255], [90, 30, 150],
                        [255, 0, 255], [255, 150, 255], [75, 0, 75], [75, 0, 175],
                        [0, 200, 255], [50, 120, 255], [0, 175, 0], [0, 60, 135],
                        [80, 240, 150], [150, 240, 255], [0, 0, 255]],  
        'classes'       : ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 
                            'road', 'parking', 'sidewalk', 'other-ground', 
                            'building', 'fence', 'vegetation', 'trunk',
                            'terrain', 'pole', 'traffic-sign']
    }
    n_class = len(SEG_CLASSES['colors'])
    
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

    #config polarseg
    ignore_label = 0
    grid_size = np.asarray([480,360,32])
    max_volume_space = np.asarray([50,np.pi,1.5])
    min_volume_space = np.asarray([3,-np.pi,-3])
    intervals = (max_volume_space - min_volume_space) / (grid_size-1)
    #untuk operasi langsung tensor
    grid_size_ten = torch.from_numpy(np.asarray([480,360,32])).to(gpu_device, dtype=dtype)
    max_volume_space_ten = torch.from_numpy(np.asarray([50,np.pi,1.5])).to(gpu_device, dtype=dtype)
    min_volume_space_ten = torch.from_numpy(np.asarray([3,-np.pi,-3])).to(gpu_device, dtype=dtype)
    intervals_ten = (max_volume_space_ten - min_volume_space_ten) / (grid_size_ten-1)

    #untuk front_dep dan bev_dep
    #100 untuk HDL32E, 200 untuk VLP32C
    # dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
    dep_min = cover_area_f[0]

    #other, buat join_img dll
    fps = 20
    rgb_res_ori = [720, 1280] #HxW
    scale_w = rgb_res_ori[1]/front_w
    scaled_H_rgb = int(rgb_res_ori[0]/scale_w)

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
