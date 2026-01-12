from collections import deque
import sys
import numpy as np
from torch import torch, cat, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms




#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_init(m):
    # print(m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1, dilationx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros', dilation=dilationx)
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, kernel, padding, dilation): #, final=False, 
        super(ConvBlock, self).__init__()
        #conv block
        self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], kernelx=kernel, paddingx=padding, dilationx=dilation)
        self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], kernelx=kernel, paddingx=padding, dilationx=dilation)
        #init
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)
 
    def forward(self, x):
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control



class xr20(nn.Module): #
    def __init__(self, config):
        super(xr20, self).__init__()
        self.config = config
        if config.inputs == 'segdep':
            in_ch = config.n_class + 1
        elif config.inputs == 'seg':
            in_ch = config.n_class
        else: #dep
            in_ch = 1
        #------------------------------------------------------------------------------------------------
        #BEV Decoder
        self.bev_cb0 = ConvBlock(channel=[in_ch, config.n_fmap[0]], kernel=5, padding=3, dilation=2)
        self.bev_pool0 = nn.AvgPool2d(kernel_size=[2,2]) #H,W
        self.bev_cb1 = ConvBlock(channel=[config.n_fmap[0], config.n_fmap[1]], kernel=3, padding=2, dilation=2)
        self.bev_pool1 = nn.AvgPool2d(kernel_size=[4,4]) #H,W
        self.bev_cb2 = ConvBlock(channel=[config.n_fmap[1], config.n_fmap[2]], kernel=3, padding=1, dilation=1)
        self.bev_pool2 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        self.bev_cb3 = ConvBlock(channel=[config.n_fmap[2], config.n_fmap[3]], kernel=3, padding=1, dilation=1)
        self.bev_pool3 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        #------------------------------------------------------------------------------------------------
        #FRONT Decoder
        self.fro_cb0 = ConvBlock(channel=[in_ch, config.n_fmap[0]], kernel=5, padding=3, dilation=2)
        self.fro_pool0 = nn.AvgPool2d(kernel_size=[2,4]) #H,W
        self.fro_cb1 = ConvBlock(channel=[config.n_fmap[0], config.n_fmap[1]], kernel=3, padding=2, dilation=2)
        self.fro_pool1 = nn.AvgPool2d(kernel_size=[2,4]) #H,W
        self.fro_cb2 = ConvBlock(channel=[config.n_fmap[1], config.n_fmap[2]], kernel=3, padding=1, dilation=1)
        self.fro_pool2 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        self.fro_cb3 = ConvBlock(channel=[config.n_fmap[2], config.n_fmap[3]], kernel=3, padding=1, dilation=1)
        self.fro_pool3 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        #------------------------------------------------------------------------------------------------
        #feature fusion
        self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(2*config.n_fmap[3], config.n_fmap[3], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap[3], config.n_fmap[2])
        )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 8 karena concat dari wp xy, rp1 xy, rp2 xy, dan velocity lr
        self.gru = nn.GRUCell(input_size=8, hidden_size=config.n_fmap[2])
        self.pred_dwp = nn.Sequential( 
            nn.Linear(config.n_fmap[2], config.n_fmap[1]),
            nn.Linear(config.n_fmap[1], 2) #x dan y
        ) #nn.Linear(config.n_fmap[2], 2)
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        #------------------------------------------------------------------------------------------------
        ##MLP Controller ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        self.ctrl_branch = nn.ModuleList([nn.Sequential( 
            nn.Linear(config.n_fmap[2], config.n_fmap[1]),
            nn.Linear(config.n_fmap[1], 2), #str dan thrt
            nn.ReLU()
        ) for _ in range(config.n_cmd)]) #.to(self.gpu_device, dtype=torch.float)
        

    def forward(self, bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo_in, cmd):#, gt_ss):
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        BEV_features_sum = 0
        FRO_features_sum = 0
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            #tentukan input
            if self.config.inputs == 'segdep':
                bev_in = cat([bev_segs[i], bev_deps[i]], dim=1)
                fro_in = cat([front_segs[i], front_deps[i]], dim=1)
            elif self.config.inputs == 'seg':
                bev_in = bev_segs[i]
                fro_in = front_segs[i]
            else: #dep
                bev_in = bev_deps[i]
                fro_in = front_deps[i]
            
            #encode bev
            bev0 = self.bev_pool0(self.bev_cb0(bev_in))
            bev1 = self.bev_pool1(self.bev_cb1(bev0))
            bev2 = self.bev_pool2(self.bev_cb2(bev1))
            bev3 = self.bev_pool3(self.bev_cb3(bev2))
            BEV_features_sum += bev3

            #encode front
            fro0 = self.fro_pool0(self.fro_cb0(fro_in))
            fro1 = self.fro_pool1(self.fro_cb1(fro0))
            fro2 = self.fro_pool2(self.fro_cb2(fro1))
            fro3 = self.fro_pool3(self.fro_cb3(fro2))
            FRO_features_sum += fro3

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(cat([BEV_features_sum, FRO_features_sum], dim=1))
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.config.gpu_device, dtype=self.config.dtype)
        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx) 
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)
        #------------------------------------------------------------------------------------------------
        #control decoder #cmd ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        #sementara ini terpaksa loop sepanjang batch dulu, ga tau caranya supaya langsung
        # print(cmd)
        # print(len(cmd))
        # print(cmd.shape)
        control_pred = self.ctrl_branch[cmd[0].item()](hx[0:1,:])
        for i in range(1, len(cmd)): 
            # print("-----------")
            # print(cmd[i].item())
            # print(hx[i:i+1,:].shape)
            control_pred = cat([control_pred, self.ctrl_branch[cmd[i].item()](hx[i:i+1,:])], dim=0) #concat di axis batch
            # print(control_pred.shape)
        #denormalisasi
        # print(control_pred.shape)
        steering = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle

        return pred_wp, steering, throttle


    def mlp_pid_control(self, pwaypoints, angular_velo, psteer, pthrottle):
        assert(pwaypoints.size(0)==1)
        waypoints = pwaypoints[0].data.cpu().numpy()
        
        #vehicular controls dari PID
        aim_point = (waypoints[1] + waypoints[0]) / 2.0 #tengah2nya wp0 dan wp1
        #90 deg ke kanan adalah 0 radian, 90 deg ke kiri adalah 1*pi radian
        angle_rad = np.clip(np.arctan2(aim_point[1], aim_point[0]), 0, np.pi) #arctan y/x
        angle_deg = np.degrees(angle_rad)
        #ke kiri adalah 0 -> +1 == 90 -> 180, ke kanan adalah 0 -> -1 == 90 -> 0
        error_angle = (angle_deg - 90.0) * self.config.err_angle_mul
        pid_steering = self.turn_controller.step(error_angle)
        pid_steering = np.clip(pid_steering, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * self.config.des_speed_mul
        linear_velo = np.mean(angular_velo) * self.config.wheel_radius
        #delta = np.clip(desired_speed - linear_velo, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(desired_speed - linear_velo)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)

        #proses vehicular controls dari MLP
        mlp_steering = np.clip(psteer.cpu().data.numpy(), -1.0, 1.0)
        mlp_throttle = np.clip(pthrottle.cpu().data.numpy(), 0.0, self.config.max_throttle)

        #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
        act_pid_throttle = pid_throttle >= self.config.min_act_thrt
        act_mlp_throttle = mlp_throttle >= self.config.min_act_thrt
        if act_pid_throttle and act_mlp_throttle:
            act_pid_steering = np.abs(pid_steering) >= self.config.min_act_thrt
            act_mlp_steering = np.abs(mlp_steering) >= self.config.min_act_thrt
            if act_pid_steering and not act_mlp_steering:
                steering = pid_steering
            elif act_mlp_steering and not act_pid_steering:
                steering = mlp_steering
            else: #keduanya sama2 kurang dari threshold atau sama2 lebih dari threshold
                steering = self.config.cw_pid[0]*pid_steering + self.config.cw_mlp[0]*mlp_steering
            throttle = self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle
        elif act_pid_throttle and not act_mlp_throttle:
            steering = pid_steering
            throttle = pid_throttle
        elif act_mlp_throttle and not act_pid_throttle:
            steering = mlp_steering
            throttle = mlp_throttle
        else: # (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
            steering = 0.0 #dinetralkan
            throttle = 0.0
        steering = float(steering)
        throttle = float(throttle)

        # print(waypoints[2])
        
        metadata = {
            'lr_velo': [float(angular_velo[0]), float(angular_velo[1])],
            'linear_velo' : float(linear_velo),
            'steering': steering,
            'throttle': throttle,
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1])],
            'pid_steering': float(pid_steering),
            'pid_throttle': float(pid_throttle),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1])],
            'mlp_steering': float(mlp_steering),
            'mlp_throttle': float(mlp_throttle),
            'wp_3': [float(waypoints[2][0].astype(np.float64)), float(waypoints[2][1].astype(np.float64))], #tambahan
            'wp_2': [float(waypoints[1][0].astype(np.float64)), float(waypoints[1][1].astype(np.float64))],
            'wp_1': [float(waypoints[0][0].astype(np.float64)), float(waypoints[0][1].astype(np.float64))],
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle_deg.astype(np.float64)),
            'aim': [float(aim_point[0].astype(np.float64)), float(aim_point[1].astype(np.float64))],
            # 'delta': float(delta.astype(np.float64)),
            'robot_pos_global': None, #akan direplace nanti
            'robot_bearing': None,
            'rp1_pos_global': None, #akan direplace nanti
            'rp2_pos_global': None, #akan direplace nanti
            'rp1_pos_local': None, #akan direplace nanti
            'rp2_pos_local': None, #akan direplace nanti
            'cmd': None, #akan direplace nanti
            'fps': None,
            'model_fps': None,
            'intervention': False,
        }
        return steering, throttle, metadata





class xr20_seg_seq1(nn.Module): #
    def __init__(self, config):
        super(xr20_seg_seq1, self).__init__()
        self.config = config
        #------------------------------------------------------------------------------------------------
        #BEV Decoder
        self.bev_cb0 = ConvBlock(channel=[config.n_class, config.n_fmap[0]], kernel=5, padding=3, dilation=2)
        self.bev_pool0 = nn.AvgPool2d(kernel_size=[2,2]) #H,W
        self.bev_cb1 = ConvBlock(channel=[config.n_fmap[0], config.n_fmap[1]], kernel=3, padding=2, dilation=2)
        self.bev_pool1 = nn.AvgPool2d(kernel_size=[4,4]) #H,W
        self.bev_cb2 = ConvBlock(channel=[config.n_fmap[1], config.n_fmap[2]], kernel=3, padding=1, dilation=1)
        self.bev_pool2 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        self.bev_cb3 = ConvBlock(channel=[config.n_fmap[2], config.n_fmap[3]], kernel=3, padding=1, dilation=1)
        self.bev_pool3 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        #------------------------------------------------------------------------------------------------
        #FRONT Decoder
        self.fro_cb0 = ConvBlock(channel=[config.n_class, config.n_fmap[0]], kernel=5, padding=3, dilation=2)
        self.fro_pool0 = nn.AvgPool2d(kernel_size=[2,4]) #H,W
        self.fro_cb1 = ConvBlock(channel=[config.n_fmap[0], config.n_fmap[1]], kernel=3, padding=2, dilation=2)
        self.fro_pool1 = nn.AvgPool2d(kernel_size=[2,4]) #H,W
        self.fro_cb2 = ConvBlock(channel=[config.n_fmap[1], config.n_fmap[2]], kernel=3, padding=1, dilation=1)
        self.fro_pool2 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        self.fro_cb3 = ConvBlock(channel=[config.n_fmap[2], config.n_fmap[3]], kernel=3, padding=1, dilation=1)
        self.fro_pool3 = nn.MaxPool2d(kernel_size=[2,2]) #H,W
        #------------------------------------------------------------------------------------------------
        #feature fusion
        self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(2*config.n_fmap[3], config.n_fmap[3], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap[3], config.n_fmap[2])
        )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 8 karena concat dari wp xy, rp1 xy, rp2 xy, dan velocity lr
        self.gru = nn.GRUCell(input_size=8, hidden_size=config.n_fmap[2])
        self.pred_dwp = nn.Sequential( 
            nn.Linear(config.n_fmap[2], config.n_fmap[1]),
            nn.Linear(config.n_fmap[1], 2) #x dan y
        ) #nn.Linear(config.n_fmap[2], 2)
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        #------------------------------------------------------------------------------------------------
        ##MLP Controller ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        self.ctrl_branch = nn.ModuleList([nn.Sequential( 
            nn.Linear(config.n_fmap[2], config.n_fmap[1]),
            nn.Linear(config.n_fmap[1], 2), #str dan thrt
            nn.ReLU()
        ) for _ in range(config.n_cmd)]) #.to(self.gpu_device, dtype=torch.float)
        

    def forward(self, bev_seg, front_seg, rp1, rp2, velo_in, cmd):#, gt_ss):
        #------------------------------------------------------------------------------------------------
        #encode bev
        bev0 = self.bev_pool0(self.bev_cb0(bev_seg))
        bev1 = self.bev_pool1(self.bev_cb1(bev0))
        bev2 = self.bev_pool2(self.bev_cb2(bev1))
        bev3 = self.bev_pool3(self.bev_cb3(bev2))
        #encode front
        fro0 = self.fro_pool0(self.fro_cb0(front_seg))
        fro1 = self.fro_pool1(self.fro_cb1(fro0))
        fro2 = self.fro_pool2(self.fro_cb2(fro1))
        fro3 = self.fro_pool3(self.fro_cb3(fro2))

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(cat([bev3, fro3], dim=1))
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(1, 2)).to(self.config.gpu_device, dtype=self.config.dtype)
        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx) 
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)
        #------------------------------------------------------------------------------------------------
        #control decoder #cmd ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        #sementara ini terpaksa loop sepanjang batch dulu, ga tau caranya supaya langsung
        # print(cmd)
        # print(len(cmd))
        # print(cmd.shape)
        control_pred = self.ctrl_branch[cmd](hx)
        #denormalisasi
        # print(control_pred.shape)
        steering = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle

        return pred_wp, steering, throttle


    def mlp_pid_control(self, pwaypoints, angular_velo, psteer, pthrottle):
        assert(pwaypoints.size(0)==1)
        waypoints = pwaypoints[0].data.cpu().numpy()
        
        #vehicular controls dari PID
        aim_point = (waypoints[1] + waypoints[0]) / 2.0 #tengah2nya wp0 dan wp1
        #90 deg ke kanan adalah 0 radian, 90 deg ke kiri adalah 1*pi radian
        angle_rad = np.clip(np.arctan2(aim_point[1], aim_point[0]), 0, np.pi) #arctan y/x
        angle_deg = np.degrees(angle_rad)
        #ke kiri adalah 0 -> +1 == 90 -> 180, ke kanan adalah 0 -> -1 == 90 -> 0
        error_angle = (angle_deg - 90.0) * self.config.err_angle_mul
        pid_steering = self.turn_controller.step(error_angle)
        pid_steering = np.clip(pid_steering, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * self.config.des_speed_mul
        linear_velo = np.mean(angular_velo) * self.config.wheel_radius
        #delta = np.clip(desired_speed - linear_velo, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(desired_speed - linear_velo)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)

        #proses vehicular controls dari MLP
        mlp_steering = np.clip(psteer.cpu().data.numpy(), -1.0, 1.0)
        mlp_throttle = np.clip(pthrottle.cpu().data.numpy(), 0.0, self.config.max_throttle)

        #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
        act_pid_throttle = pid_throttle >= self.config.min_act_thrt
        act_mlp_throttle = mlp_throttle >= self.config.min_act_thrt
        if act_pid_throttle and act_mlp_throttle:
            act_pid_steering = np.abs(pid_steering) >= self.config.min_act_thrt
            act_mlp_steering = np.abs(mlp_steering) >= self.config.min_act_thrt
            if act_pid_steering and not act_mlp_steering:
                steering = pid_steering
            elif act_mlp_steering and not act_pid_steering:
                steering = mlp_steering
            else: #keduanya sama2 kurang dari threshold atau sama2 lebih dari threshold
                steering = self.config.cw_pid[0]*pid_steering + self.config.cw_mlp[0]*mlp_steering
            throttle = self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle
        elif act_pid_throttle and not act_mlp_throttle:
            steering = pid_steering
            throttle = pid_throttle
        elif act_mlp_throttle and not act_pid_throttle:
            steering = mlp_steering
            throttle = mlp_throttle
        else: # (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
            steering = 0.0 #dinetralkan
            throttle = 0.0
        steering = float(steering)
        throttle = float(throttle)

        # print(waypoints[2])
        
        metadata = {
            'lr_velo': [float(angular_velo[0]), float(angular_velo[1])],
            'linear_velo' : float(linear_velo),
            'steering': steering,
            'throttle': throttle,
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1])],
            'pid_steering': float(pid_steering),
            'pid_throttle': float(pid_throttle),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1])],
            'mlp_steering': float(mlp_steering),
            'mlp_throttle': float(mlp_throttle),
            'wp_3': [float(waypoints[2][0].astype(np.float64)), float(waypoints[2][1].astype(np.float64))], #tambahan
            'wp_2': [float(waypoints[1][0].astype(np.float64)), float(waypoints[1][1].astype(np.float64))],
            'wp_1': [float(waypoints[0][0].astype(np.float64)), float(waypoints[0][1].astype(np.float64))],
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle_deg.astype(np.float64)),
            'aim': [float(aim_point[0].astype(np.float64)), float(aim_point[1].astype(np.float64))],
            # 'delta': float(delta.astype(np.float64)),
            'robot_pos_global': None, #akan direplace nanti
            'robot_bearing': None,
            'rp1_pos_global': None, #akan direplace nanti
            'rp2_pos_global': None, #akan direplace nanti
            'rp1_pos_local': None, #akan direplace nanti
            'rp2_pos_local': None, #akan direplace nanti
            'cmd': None, #akan direplace nanti
            'fps': None,
            'model_fps': None,
            'intervention': False,
        }
        return steering, throttle, metadata

