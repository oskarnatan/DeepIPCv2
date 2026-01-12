import os
import yaml
import cv2
# from PIL import Image, ImageFile
from collections import deque
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from pypcd import pypcd #https://github.com/dimatura/pypcd/issues/7 #pip3 install --upgrade git+https://github.com/klintan/pypcd.git 

import numpy as np
import torch 
from torch.utils.data import Dataset


class WHILL_Data(Dataset):
    def __init__(self, data_root, conditions, config):
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.data_rate = config.hz
        self.rp1_close = config.rp1_close

        self.condition = [] #buat offline test nantinya
        self.route = []
        self.filename = []
        self.rgb = []
        self.dep_cld_xyz = []
        # self.seg = []
        self.pt_cld_pcd = []
        self.pt_cld_seg = []
        self.lon = []
        self.lat = []
        self.loc_x = []
        self.loc_y = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.loc_heading = []
        self.steering = []
        self.throttle = []
        self.velocity_l = []
        self.velocity_r = []
        
        for condition in conditions:
            sub_root = os.path.join(data_root, condition)
            preload_file = os.path.join(sub_root, 'seq'+str(self.seq_len)+'_pred'+str(self.pred_len)+'.npy') #'_rp1'+str(self.rp1_close)+

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_condition = []
                preload_route = []
                preload_filename = []
                preload_rgb = []
                preload_dep_cld_xyz = []
                # preload_seg = []
                preload_pt_cld_pcd = []
                preload_pt_cld_seg = []
                preload_lon = []
                preload_lat = []
                preload_loc_x = []
                preload_loc_y = []
                preload_rp1_lon = []
                preload_rp1_lat = []
                preload_rp2_lon = []
                preload_rp2_lat = []
                preload_bearing = []
                preload_loc_heading = []
                preload_steering = []
                preload_throttle = []
                preload_velocity_l = []
                preload_velocity_r = []
                
                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                root_files.sort() #nanti sudah diacak oleh torch dataloader
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    
                    #load route list nya
                    with open(route_dir+"/"+route+"_routepoint_list.yml", 'r') as rp_listx:
                    # with open(route_dir+"/gmaps"+route[-2:]+"_routepoint_list.yml", 'r') as rp_listx:
                        rp_list = yaml.safe_load(rp_listx)
                        #assign end point sebagai route terakhir
                        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
                    
                    #list dan sort file, slah satu saja
                    files = os.listdir(route_dir+"/meta/")
                    files.sort() #nanti sudah diacak oleh torch dataloader

                    for i in range(0, len(files)-(self.seq_len-1)-(self.pred_len*self.data_rate)): #kurangi sesuai dengan jumlah sequence dan wp yang akan diprediksi
                        #ini yang buat yg disequence kan
                        rgbs = []
                        dep_cld_xyzs = []
                        # segs = []
                        pt_cld_pcds = []
                        pt_cld_segs = []
                        loc_xs = []
                        loc_ys = []
                        loc_headings = []
                        
                        # read files sequentially (past and current frames)
                        for j in range(0, self.seq_len):
                            filename = files[i+j][:-3] #hilangkan ekstensi filenya
                            rgbs.append(route_dir+"/stereolabs/rgb/"+filename+"png")
                            dep_cld_xyzs.append(route_dir+"/stereolabs/depth/depth_cloud/"+filename+"npy")
                            # segs.append(route_dir+"/segmentation_GT/"+filename)
                            # pt_clouds.append(route_dir+"/point_cloud/"+filename+"npy")
                            pt_cld_pcds.append(route_dir+"/velodyne/lidar/"+filename+"pcd")
                            pt_cld_segs.append(route_dir+"/velodyne/lidseg/"+filename+"npy")

                        #appendkan
                        preload_rgb.append(rgbs)
                        preload_dep_cld_xyz.append(dep_cld_xyzs)
                        # preload_seg.append(segs)
                        preload_pt_cld_pcd.append(pt_cld_pcds)
                        preload_pt_cld_seg.append(pt_cld_segs)

                        #metadata buat testing nantinya
                        preload_condition.append(condition)
                        preload_route.append(route)
                        preload_filename.append(filename)

                        # ambil local loc, heading, vehicular controls, gps loc, dan bearing pada seq terakhir saja (current)
                        with open(route_dir+"/meta/"+filename+"yml", "r") as read_meta_current:
                            meta_current = yaml.safe_load(read_meta_current)
                        loc_xs.append(meta_current['whill_local_position_xyz'][0])
                        loc_ys.append(meta_current['whill_local_position_xyz'][1])
                        loc_headings.append(np.radians(meta_current['whill_local_orientation_rpy'][2]))
                        preload_lon.append(meta_current['ublox_longitude'])
                        preload_lat.append(meta_current['ublox_latitude'])

                        #bias bearing
                        bearing_robot_deg = bearing_biasing(meta_current['wit_EKF_rpy'][2], config.bearing_bias)
                        preload_bearing.append(np.radians(bearing_robot_deg))

                        #vehicular controls
                        preload_steering.append(meta_current['whill_steering'])
                        preload_throttle.append(meta_current['whill_throttle'])
                        preload_velocity_l.append(np.abs(meta_current['whill_LR_wheel_angular_velo'][0])) #kecepatan LR dibuat positif semua
                        preload_velocity_r.append(np.abs(meta_current['whill_LR_wheel_angular_velo'][1])) #kecepatan LR dibuat positif semua

                        
                        #assign next route lat lon
                        about_to_finish = False
                        for r in range(2): #ada 2 route point
                            next_lat = rp_list['route_point']['latitude'][r]
                            next_lon = rp_list['route_point']['longitude'][r]
                            dLat_m = (next_lat-meta_current['ublox_latitude']) * 40008000 / 360 #111320 #Y
                            dLon_m = (next_lon-meta_current['ublox_longitude']) * 40075000 * np.cos(np.radians(meta_current['ublox_latitude'])) / 360 #X
                            
                            if r==0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close and not about_to_finish: #jika jarak euclidian rp1 <= jarak min, hapus route dan loncat ke next route
                                if len(rp_list['route_point']['latitude']) > 2: #jika jumlah route list masih > 2
                                    rp_list['route_point']['latitude'].pop(0)
                                    rp_list['route_point']['longitude'].pop(0)
                                else: #berarti mendekati finish
                                    about_to_finish = True
                                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                                next_lat = rp_list['route_point']['latitude'][r]
                                next_lon = rp_list['route_point']['longitude'][r]
                            
                            if r==0:
                                preload_rp1_lon.append(next_lon)
                                preload_rp1_lat.append(next_lat)
                            else: #r==1
                                preload_rp2_lon.append(next_lon)
                                preload_rp2_lat.append(next_lat)


                        # read files sequentially (future frames)
                        for k in range(1, self.pred_len+1):
                            filenamef = files[(i+self.seq_len-1) + (k*self.data_rate)] #future seconds, makanya dikali data rate
                            # meta
                            with open(route_dir+"/meta/"+filenamef, "r") as read_meta_future: #+"yml"
                                meta_future = yaml.safe_load(read_meta_future)
                            loc_xs.append(meta_future['whill_local_position_xyz'][0])
                            loc_ys.append(meta_future['whill_local_position_xyz'][1])
                            loc_headings.append(np.radians(meta_future['whill_local_orientation_rpy'][2]))

                        #append sisanya
                        preload_loc_x.append(loc_xs)
                        preload_loc_y.append(loc_ys)
                        preload_loc_heading.append(loc_headings)


                # dump ke npy
                preload_dict = {}
                preload_dict['condition'] = preload_condition
                preload_dict['route'] = preload_route
                preload_dict['filename'] = preload_filename
                preload_dict['rgb'] = preload_rgb
                preload_dict['dep_cld_xyz'] = preload_dep_cld_xyz
                # preload_dict['seg'] = preload_seg
                preload_dict['pt_cld_pcd'] = preload_pt_cld_pcd
                preload_dict['pt_cld_seg'] = preload_pt_cld_seg
                preload_dict['lon'] = preload_lon
                preload_dict['lat'] = preload_lat
                preload_dict['loc_x'] = preload_loc_x
                preload_dict['loc_y'] = preload_loc_y
                preload_dict['rp1_lon'] = preload_rp1_lon
                preload_dict['rp1_lat'] = preload_rp1_lat
                preload_dict['rp2_lon'] = preload_rp2_lon
                preload_dict['rp2_lat'] = preload_rp2_lat
                preload_dict['bearing'] = preload_bearing
                preload_dict['loc_heading'] = preload_loc_heading
                preload_dict['steering'] = preload_steering
                preload_dict['throttle'] = preload_throttle
                preload_dict['velocity_l'] = preload_velocity_l
                preload_dict['velocity_r'] = preload_velocity_r
                np.save(preload_file, preload_dict)


            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.condition += preload_dict.item()['condition']
            self.route += preload_dict.item()['route']
            self.filename += preload_dict.item()['filename']
            self.rgb += preload_dict.item()['rgb']
            self.dep_cld_xyz += preload_dict.item()['dep_cld_xyz']
            # self.seg += preload_dict.item()['seg']
            self.pt_cld_pcd += preload_dict.item()['pt_cld_pcd']
            self.pt_cld_seg += preload_dict.item()['pt_cld_seg']
            self.lon += preload_dict.item()['lon']
            self.lat += preload_dict.item()['lat']
            self.loc_x += preload_dict.item()['loc_x']
            self.loc_y += preload_dict.item()['loc_y']
            self.rp1_lon += preload_dict.item()['rp1_lon']
            self.rp1_lat += preload_dict.item()['rp1_lat']
            self.rp2_lon += preload_dict.item()['rp2_lon']
            self.rp2_lat += preload_dict.item()['rp2_lat']
            self.bearing += preload_dict.item()['bearing']
            self.loc_heading += preload_dict.item()['loc_heading']
            self.steering += preload_dict.item()['steering']
            self.throttle += preload_dict.item()['throttle']
            self.velocity_l += preload_dict.item()['velocity_l']
            self.velocity_r += preload_dict.item()['velocity_r']
            print("Preloading " + str(len(preload_dict.item()['pt_cld_pcd'])) + " sequences from " + preload_file)

    def __len__(self):
        # return len(self.rgb)
        return len(self.pt_cld_pcd)

    def __getitem__(self, index):
        data = dict()
        #metadata buat testing nantinya
        data['condition'] = self.condition[index]
        data['route'] = self.route[index]
        data['filename'] = self.filename[index]

        # data['rgbs'] = []
        # data['segs'] = []
        # data['ptxs'] = []
        # data['ptys'] = []
        # data['ptzs'] = []
        # data['ptss'] = []
        # data['velodyne'] = []
        # data['polarnet'] = []
        data['bev_segs'] = []
        data['bev_deps'] = []
        data['front_segs'] = []
        data['front_deps'] = []
        # seq_rgbs = self.rgb[index]
        # seq_segs = self.seg[index]
        # seq_pt_clouds = self.pt_cloud[index]
        seq_pt_cloud_pcds = self.pt_cld_pcd[index]
        seq_pt_cloud_segs = self.pt_cld_seg[index]
        seq_loc_xs = self.loc_x[index]
        seq_loc_ys = self.loc_y[index]
        seq_loc_headings = self.loc_heading[index]

        for i in range(0, self.seq_len):
            # data['rgbs'].append(torch.from_numpy(np.array(crop_matrix(cv2.imread(seq_rgbs[i]), resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1))))
            # data['segs'].append(torch.from_numpy(np.array(cls2one_hot(crop_matrix(cv2.imread(seq_segs[i]), resize=self.config.scale, crop=self.config.crop_roi), n_class=self.config.n_class))))

            # pt_cloud = np.nan_to_num(crop_matrix(np.load(seq_pt_clouds[i])[:,:,0:3], resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1), nan=0.0, posinf=39.99999, neginf=0.2) #min_d, max_d, -max_d, ambil xyz-nya saja 0:3, baca https://www.stereolabs.com/docs/depth-sensing/depth-settings/
            # data['pt_cloud_xs'].append(torch.from_numpy(np.array(pt_cloud[0:1,:,:])))
            # data['pt_cloud_zs'].append(torch.from_numpy(np.array(pt_cloud[2:3,:,:])))
            lid_pc = pypcd.PointCloud.from_path(seq_pt_cloud_pcds[i])
            # lid_x = lid_pc.pc_data['x']
            # print(lid_x.shape)
            # print(lid_x)
            # lid_y = lid_pc.pc_data['y']
            # lid_z = lid_pc.pc_data['z']
            # lid_intensity = lid_pc.pc_data['intensity']
            # in_velodyne = np.zeros(lid_x.shape[0] + lid_y.shape[0] + lid_z.shape[0] + lid_intensity.shape[0], dtype=np.float32)
            # in_velodyne[0::4] = lid_x
            # in_velodyne[1::4] = lid_y
            # in_velodyne[2::4] = lid_z
            # in_velodyne[3::4] = lid_intensity
            # in_velodyne = in_velodyne.astype('float32').reshape((-1, 4))
            #in_velodyne = np.fromfile(seq_pt_cloud_pcds[i], dtype=np.float32).reshape((-1, 4))#.transpose(1,0)
            pred_lidseg = np.load(seq_pt_cloud_segs[i])#.ravel() #class id 1 - 19
            # data['ptxs'].append(in_velodyne[:,0])
            # data['ptys'].append(in_velodyne[:,2])
            # data['ptzs'].append(in_velodyne[:,1])
            # data['ptss'].append(pred_lidseg[:,0])
            # data['velodyne'].append(in_velodyne[0:3,:])
            # data['polarnet'].append(pred_lidseg) #[:,0]
            # ptx_ten = torch.tensor(in_velodyne[:,0]).to(self.config.gpu_device, dtype=self.config.dtype)
            # pty_ten = torch.tensor(in_velodyne[:,2]).to(self.config.gpu_device, dtype=self.config.dtype)
            # ptz_ten = torch.tensor(in_velodyne[:,1]).to(self.config.gpu_device, dtype=self.config.dtype)
            # ptseg_ten = torch.tensor(pred_lidseg[:,0]).to(self.config.gpu_device, dtype=self.config.dtype)
            #VELODYNE VLP32C / SEMANTIC KITTI --> xyz = 120 dan x dikali dengan -1
            #VELODYNE HDL32E --> xyz = 021
            if self.config.lidar_sensor == "vlp32c":
                ptx_arr = np.array(lid_pc.pc_data['y']) * -1 #np.array(in_velodyne[:,1]) * -1
                pty_arr = np.array(lid_pc.pc_data['z']) #np.array(in_velodyne[:,2])
                ptz_arr = np.array(lid_pc.pc_data['x']) #np.array(in_velodyne[:,0])
            else: #hdl32e
                # print(in_velodyne[:,0].shape)
                # print(in_velodyne[:,0])
                ptx_arr = np.array(lid_pc.pc_data['x']) #np.array(in_velodyne[:,0])
                pty_arr = np.array(lid_pc.pc_data['z']) #np.array(in_velodyne[:,2])
                ptz_arr = np.array(lid_pc.pc_data['y']) #np.array(in_velodyne[:,1])
            ptseg_arr = np.array(pred_lidseg[:,0])
            #pakai yg versi numpy karena tidak bisa re-initiate cuda di sini
            bev_seg, bev_dep, front_seg, front_dep = gen_bev_front_seg_dep_numpy(self.config, ptx_arr, pty_arr, ptz_arr, ptseg_arr, bs=1)
            data['bev_segs'].append(bev_seg[0])
            data['bev_deps'].append(bev_dep[0])
            data['front_segs'].append(front_seg[0])
            data['front_deps'].append(front_dep[0])

            #check
            # bev_segcol = colorize_seg(bev_seg, self.config.SEG_CLASSES['colors'])
            # bev_depcol = colorize_logdepth(bev_dep)
            # front_segcol = colorize_seg(front_seg, self.config.SEG_CLASSES['colors'])  
            # front_depcol = colorize_logdepth(front_dep) 
            # cv2.imwrite(self.config.logdir+'/'+data['filename'][:-4]+"_bevseg.png", bev_segcol)
            # cv2.imwrite(self.config.logdir+'/'+data['filename'][:-4]+"_bevdep.png", bev_depcol)
            # cv2.imwrite(self.config.logdir+'/'+data['filename'][:-4]+"_froseg.png", front_segcol)
            # cv2.imwrite(self.config.logdir+'/'+data['filename'][:-4]+"_frodep.png", front_depcol)


        #current ego robot position dan heading di index 0
        ego_loc_x = seq_loc_xs[0]
        ego_loc_y = seq_loc_ys[0]
        ego_loc_heading = seq_loc_headings[0]   

        # waypoint processing to local coordinates
        data['waypoints'] = [] #wp dalam local coordinate
        for j in range(1, self.pred_len+1):
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_loc_headings[j], seq_loc_xs[j], seq_loc_ys[j], np.pi/2-ego_loc_heading, ego_loc_x, ego_loc_y)
            data['waypoints'].append(tuple(local_waypoint[0,:2]))
      

        # convert rp1_lon, rp1_lat rp2_lon, rp2_lat ke local coordinates
        #komputasi dari global ke local
        #https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([[np.cos(bearing_robot), -np.sin(bearing_robot)],
                            [np.sin(bearing_robot),  np.cos(bearing_robot)]])
        dLat1_m = (self.rp1_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon1_m = (self.rp1_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        dLat2_m = (self.rp2_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon2_m = (self.rp2_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        rp1_bev = R_matrix.T.dot(np.array([dLon1_m, dLat1_m]))
        rp2_bev = R_matrix.T.dot(np.array([dLon2_m, dLat2_m]))
        data['rp1'] = tuple(rp1_bev)
        data['rp2'] = tuple(rp2_bev)
        data['lat_rp1'] = self.rp1_lat[index]
        data['lat_rp2'] = self.rp2_lat[index]
        data['lon_rp1'] = self.rp1_lon[index]
        data['lon_rp2'] = self.rp2_lon[index]
        #nentukan high nav command, buat seleksi controller branch nantinya
        if rp1_bev[0] >= self.config.rp1_close or rp2_bev[0] >= 2*self.config.rp1_close:
            data['cmd'] = 2 #kanan
        elif rp1_bev[0] <= -1*self.config.rp1_close or rp2_bev[0] <= -2*self.config.rp1_close:
            data['cmd'] = 1 #kiri
        else:
            data['cmd'] = 0 #lurus
        # print("rp1_lat "+str(self.rp1_lat[index]))
        # print("rp2_lat "+str(self.rp2_lat[index]))
        # print("rp1_lon "+str(self.rp1_lon[index]))
        # print("rp2_lon "+str(self.rp2_lon[index]))

        #vehicular controls dan velocity jadikan satu LR
        data['steering'] = self.steering[index]
        data['throttle'] = self.throttle[index]
        data['lr_velo'] = tuple(np.array([self.velocity_l[index], self.velocity_r[index]]))

        #metadata buat testing nantinya
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data

"""
#ini untuk mengatasi shape ptcloud yang berbeda dan tidak bisa distack secara default oleh fungsi dataloader pytorch
#baca https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
def custom_collate(data):
    datax = dict()
    datax['ptxs']=np.stack([d['ptxs'] for d in data]).astype(np.float32)
    datax['ptys']=np.stack([d['ptys'] for d in data]).astype(np.float32)
    datax['ptzs']=np.stack([d['ptzs'] for d in data]).astype(np.float32)
    datax['ptss']=np.stack([d['ptss'] for d in data]).astype(np.float32)
    #lainnya sama
    datax['condition'] = [d['condition'] for d in data]
    datax['route'] = [d['route'] for d in data]
    datax['filename'] = [d['filename'] for d in data]
    datax['waypoints'] = [d['waypoints'] for d in data]
    datax['rp1'] = [d['rp1'] for d in data]
    datax['rp2'] = [d['rp2'] for d in data]
    datax['cmd'] = [d['cmd'] for d in data]
    datax['steering'] = [d['steering'] for d in data]
    datax['throttle'] = [d['throttle'] for d in data]
    datax['lr_velo'] = [d['lr_velo'] for d in data]
    datax['bearing_robot'] = [d['bearing_robot'] for d in data]
    datax['lat_robot'] = [d['lat_robot'] for d in data]
    datax['lon_robot'] = [d['lon_robot'] for d in data]
    return datax
"""


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)


def preproc_spherical(raw_data, config):
    #load binfile yang berisi xyz-intensity dan split ke xyz dan sig
    xyz = raw_data[:,:3]
    # xyz = xyz[:,[1, 0, 2]] #tidak perlu karena polarnet sudah robust terhadap rotasi berapapun
    # xyz[:,0] = -1*xyz[:,0] #tidak perlu karena polarnet sudah robust terhadap flip x, y, dan xy
    sig = np.clip(np.squeeze(raw_data[:,3])/config.max_intensity, 0.0, 1.0) #intensity harus discale/clip dalam range 0 - 1.0, baca lidarsegdep_bev_front/check.py

    # convert coordinate into polar coordinates
    xyz_pol = cart2polar(xyz)
    # get grid index
    crop_range = config.max_volume_space - config.min_volume_space
    intervals = crop_range / (config.grid_size-1)

    # if (intervals==0).any(): print("Zero interval!")
    grid_ind = (np.floor((np.clip(xyz_pol,config.min_volume_space,config.max_volume_space)-config.min_volume_space)/intervals)).astype(np.int)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + config.min_volume_space
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]), axis=1)
    return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]), axis=1)

    return [grid_ind], [return_fea]


#operasi torch tensor
def torch_cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = torch.atan2(input_xyz[:,1],input_xyz[:,0])
    return torch.stack((rho,phi,input_xyz[:,2]),dim=1)


def torch_preproc_spherical(raw_data, config):
    # convert coordinate into polar coordinates
    xyz_pol = torch_cart2polar(raw_data[:,:3])
    sig = torch.clip(raw_data[:,3]/config.max_intensity, 0.0, 1.0) #intensity harus discale/clip dalam range 0 - 1.0, baca lidarsegdep_bev_front/check.py
    # get grid index
    grid_ind = torch.floor((torch.clip(xyz_pol,config.min_volume_space_ten,config.max_volume_space_ten)-config.min_volume_space_ten)/config.intervals_ten)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind + 0.5)*config.intervals_ten + config.min_volume_space_ten
    return_xyz = xyz_pol - voxel_centers
    return_xyz = torch.cat((return_xyz,xyz_pol,raw_data[:,:2]), dim=1)
    return_fea = torch.cat((return_xyz,sig[:,None]), dim=1)

    return grid_ind.long(), return_fea




def resizecrop_matrix(image, resize=1, D3=True, crop=[512, 1024]):
    
    #resize image
    WH_resized = (int(image.shape[1]/resize), int(image.shape[0]/resize))
    resized_image = cv2.resize(image, WH_resized, interpolation=cv2.INTER_NEAREST)

    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((resized_image.shape[0]/2) - (crop[0]/2)), int((resized_image.shape[1]/2) - (crop[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else: #buat matrix 2d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]


    return cropped_im


def gen_bev_front_seg_dep_numpy(self, ptx, pty, ptz, ptseg, bs=None):
    if bs == None:
        bsize = self.batch_size
    else:
        bsize = bs
    #flatten all
    ptx = ptx.ravel()
    pty = pty.ravel()
    ptz = ptz.ravel() - self.cover_area_f[0]
    d_lidar = np.sqrt(ptx**2 + ptz**2) #jarak relatif #+ pty**2 
    ptseg = ptseg.ravel()
    ptn = np.array([[n for _ in range(len(ptseg))] for n in range(bsize)]).ravel()
    
    #check shape
    # print(ptn.shape)
    # print(ptseg.shape)

    #normalize ke frame untuk BEV projection
    frame_data_x = np.round((ptx+self.cover_area_lr) * (self.bev_w-1) / (2*self.cover_area_lr))
    frame_data_z = np.round((ptz * (1-self.bev_h) / (self.cover_area_f[1]-self.cover_area_f[0])) + (self.bev_h-1))

    #BEV SEG
    #cari index interest
    boolx = np.logical_and(frame_data_x <= self.bev_w-1, frame_data_x >= 0)
    bool_all = np.logical_and(boolx, np.logical_and(frame_data_z <= self.bev_h-1, frame_data_z >= 0))
    # print(bool_all.shape)
    idx = bool_all.nonzero()#.squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    coorx = np.stack([ptn, ptseg, frame_data_z, frame_data_x])
    coor_clsn = np.unique(coorx[:, idx], axis=1).astype(np.int64) #tensor harus long supaya bisa digunakan sebagai index
    bev_seg = np.zeros((bsize, self.n_class, self.bev_h, self.bev_w))
    bev_seg[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW


    # bev_dep
    idx_dlidar = np.argwhere(bool_all)
    bev_dep = np.zeros((bsize, 1, self.bev_h, self.bev_w))
    linear_d = np.clip(((9*(d_lidar[idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), a_min=1.0, a_max=10.0) #linear 1 - 10
    log_d = -1*np.log(linear_d) + 1 #logarithmic 1 - 0
    bev_dep[ptn[idx_dlidar].astype(np.int64), 0, frame_data_z[idx_dlidar].astype(np.int64), frame_data_x[idx_dlidar].astype(np.int64)] = log_d
    # bev_dep = bev_dep / (self.dep_max - self.dep_min)
    # print(bev_dep)


    #FRONT SEG
    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-ptz, ptx)/ self.h_res_rad
    y_img = np.arctan2(pty, d_lidar)/ self.v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / self.h_res / 2  # Theoretical min x value based on sensor specs
    x_img = x_img - x_min              # Shift

    y_min = self.v_fov[0] / self.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min             # Shift
    y_max = int(self.v_fov_total / self.v_res) # Theoretical max x value after shifting
    y_img = -1 * (y_img - y_max) #di flip

    boolx = np.logical_and(x_img <= self.front_w-1, x_img >= 0)
    bool_all = np.logical_and(boolx, np.logical_and(y_img <= self.front_h-1, y_img >= 0))
    idx = bool_all.nonzero()#.squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    coorx = np.stack([ptn, ptseg, y_img, x_img])
    coor_clsn = np.unique(coorx[:, idx], axis=1).astype(np.int64) #tensor harus int supaya bisa digunakan sebagai index
    front_seg = np.zeros((bsize, self.n_class, self.front_h, self.front_w))
    front_seg[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW

    # front_dep
    idx_dlidar = np.argwhere(bool_all)
    front_dep = np.zeros((bsize, 1, self.front_h, self.front_w))
    linear_d = np.clip(((9*(d_lidar[idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), a_min=1.0, a_max=10.0) #linear 1 - 10
    log_d = -1*np.log(linear_d) + 1 #logarithmic 1 - 0
    front_dep[ptn[idx_dlidar].astype(np.int64), 0, y_img[idx_dlidar].astype(np.int64), x_img[idx_dlidar].astype(np.int64)] = log_d
    # front_dep = front_dep / (self.dep_max - self.dep_min)
    # print(front_dep)

    return bev_seg, bev_dep, front_seg, front_dep


def gen_bev_front_seg_dep(self, ptx, pty, ptz, ptseg, bs=None, return_dep=True):
    if bs == None:
        bsize = self.batch_size
    else:
        bsize = bs
    #flatten all
    ptx = ptx.ravel()
    pty = pty.ravel()
    ptz = ptz.ravel() - self.cover_area_f[0]
    d_lidar = torch.sqrt(ptx**2 + ptz**2) #jarak relatif #+ pty**2 
    ptseg = ptseg.ravel()
    ptn = torch.ravel(torch.tensor([[n for _ in range(len(ptseg))] for n in range(bsize)])).to(self.gpu_device, dtype=self.dtype) #dummy batch
    
    #check shape
    # print(ptn.shape)
    # print(ptseg.shape)

    #normalize ke frame untuk BEV projection
    frame_data_x = torch.round((ptx+self.cover_area_lr) * (self.bev_w-1) / (2*self.cover_area_lr))
    frame_data_z = torch.round((ptz * (1-self.bev_h) / (self.cover_area_f[1]-self.cover_area_f[0])) + (self.bev_h-1))

    #BEV SEG
    #cari index interest
    boolx = torch.logical_and(frame_data_x <= self.bev_w-1, frame_data_x >= 0)
    bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_z <= self.bev_h-1, frame_data_z >= 0))
    idx = bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    coorx = torch.stack([ptn, ptseg, frame_data_z, frame_data_x])
    coor_clsn = torch.unique(coorx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    bev_seg = torch.zeros((bsize, self.n_class, self.bev_h, self.bev_w), dtype=self.dtype, device=self.gpu_device)
    bev_seg[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW


    # bev_dep
    if return_dep:
        idx_dlidar = torch.nonzero(bool_all) # atau torch.argwhere(bool_all)
        # coorxx = torch.stack([ptn, frame_data_z, frame_data_x])
        # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        bev_dep = torch.zeros((bsize, 1, self.bev_h, self.bev_w), dtype=self.dtype, device=self.gpu_device)
        linear_d = torch.clip(((9*(d_lidar[idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), min=1.0, max=10.0) #linear 1 - 10
        log_d = -1*torch.log(linear_d) + 1 #logarithmic 1 - 0
        bev_dep[ptn[idx_dlidar].long(), 0, frame_data_z[idx_dlidar].long(), frame_data_x[idx_dlidar].long()] = log_d
        # bev_dep = bev_dep / (self.dep_max - self.dep_min)
        # print(bev_dep)
    else:
        bev_dep = None


    #FRONT SEG
    #baca: https://github.com/collector-m/lidar_projection/blob/master/show.py
    # Distance relative to origin when looked from top
    # d_lidar = torch.sqrt(ptx**2 + ptz**2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)


    # PROJECT INTO IMAGE COORDINATES
    x_img = torch.atan2(-ptz, ptx)/ self.h_res_rad
    y_img = torch.atan2(pty, d_lidar)/ self.v_res_rad


    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / self.h_res / 2  # Theoretical min x value based on sensor specs
    x_img = x_img - x_min              # Shift
    # x_max = int(360.0 / self.h_res)       # Theoretical max x value after shifting

    y_min = self.v_fov[0] / self.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min             # Shift
    y_max = int(self.v_fov_total / self.v_res) # Theoretical max x value after shifting

    # y_max = int(y_max + self.y_fudge)            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.
    y_img = -1 * (y_img - y_max) #di flip

    #normalize ke frame untuk FRONT projection
    #baca https://towardsdatascience.com/spherical-projection-for-point-clouds-56a2fc258e6c
    # ptR = torch.sqrt(torch.pow(ptx,2) + torch.pow(ptx,2) + torch.pow(ptx,2))
    # pt_pitch = torch.asin(ptz/ptR)
    # pt_yaw = torch.atan2(pty,ptx)
    # frame_data_pitch = (self.front_h-1) * (1-(pt_pitch-self.fov_down)/self.fov)
    # frame_data_yaw = (self.front_w-1) * (0.5*((pt_yaw/np.pi)+1))
    # frame_data_y = torch.round(((pty-self.cover_area_up[0]) * (1-self.front_h) / (self.cover_area_up[1]-self.cover_area_up[0])) + (self.front_h-1))

    #cari index interest
    # boolxz = torch.logical_and(boolx, ptz >= self.cover_area_f[0]) #ptz >= self.cover_area_f[0] berarti point2 yang berada didepan vehicle saja
    # bool_all = torch.logical_and(boolxz, torch.logical_and(frame_data_y <= self.front_h-1, frame_data_y >= 0))
    # boolx = torch.logical_and(frame_data_yaw <= self.front_w-1, frame_data_yaw >= 0)
    # bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_pitch <= self.front_h-1, frame_data_pitch >= 0))
    boolx = torch.logical_and(x_img <= self.front_w-1, x_img >= 0)
    bool_all = torch.logical_and(boolx, torch.logical_and(y_img <= self.front_h-1, y_img >= 0))
    idx = bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    # coorx = torch.stack([ptn, ptseg, frame_data_y, frame_data_x])
    # coorx = torch.stack([ptn, ptseg, frame_data_pitch, frame_data_yaw])
    coorx = torch.stack([ptn, ptseg, y_img, x_img])
    coor_clsn = torch.unique(coorx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    # front_seg = torch.zeros((bsize, self.n_class, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
    front_seg = torch.zeros((bsize, self.n_class, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
    front_seg[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW


    # front_dep
    if return_dep:
        # front_dep = torch.zeros((bsize, 1, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
        # coorxx = torch.stack([ptn, d_lidar, y_img, x_img])
        # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        # front_dep[coor_depn[0], 0, coor_depn[2], coor_depn[3]] = d_lidar[coor_depn[1]]
        idx_dlidar = torch.nonzero(bool_all) # atau torch.argwhere(bool_all)
        # coorxx = torch.stack([ptn, frame_data_z, frame_data_x])
        # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        front_dep = torch.zeros((bsize, 1, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
        linear_d = torch.clip(((9*(d_lidar[idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), min=1.0, max=10.0) #linear 1 - 10
        log_d = -1*torch.log(linear_d) + 1 #logarithmic 1 - 0
        front_dep[ptn[idx_dlidar].long(), 0, y_img[idx_dlidar].long(), x_img[idx_dlidar].long()] = log_d
        # front_dep = front_dep / (self.dep_max - self.dep_min)
        # print(front_dep)
    else:
        front_dep = None

    return bev_seg, bev_dep, front_seg, front_dep


def colorize_seg(sem_map, colmap):
    #buat array kosong untuk menyimpan output gambar
    sem_img = np.zeros((sem_map.shape[2], sem_map.shape[3], 3))
    idx = np.argmax(sem_map[0], axis=0)
    for cmap in colmap:
        cmap_id = colmap.index(cmap)
        sem_img[np.where(idx == cmap_id)] = cmap
    # sem_img = sem_img[:, :, [2, 1, 0]]
    # print(sem_img.shape)
    return sem_img

def colorize_logdepth(depth_map):
    #inputnya sudah 0 - 1
    norm_dep = depth_map[0][0] 

    logdepth = np.ones(norm_dep.shape) + (np.log(norm_dep) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0) 
    logdepth = np.repeat(logdepth[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return logdepth



def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out



def plot_bev_rpwp(configx, local_bev_x, local_bev_y):
    #komputasi dari lokal ke bev frame buat visualisasi
    x_img = (local_bev_x+configx.cover_area_lr)*(configx.bev_w-1)/(2*configx.cover_area_lr)
    y_img = (local_bev_y*(1-configx.bev_h)/(configx.cover_area_f[1]-configx.cover_area_f[0])) + (configx.bev_h-1)

    #batasan
    x_img = np.clip(int(x_img), 0, configx.bev_w-1)#constrain
    # nextr_x_frame = np.clip(int((local_bev_x+(cover_area-cover_min)/2)*(www-1)/((cover_area-cover_min))), 0, www-1)#constrain
    y_img = np.clip(int(y_img), 0, configx.bev_h-1)#constrain

    return x_img, y_img

#baca: https://github.com/collector-m/lidar_projection/blob/master/show.py
def plot_front_rpwp(configx, local_bev_x, local_bev_y):
    y_road = configx.cover_area_up[0] #-1.5 #ketinggian jalan dari perspektif posisi LiDAR, dalam meter
    xy_euclid = np.sqrt(local_bev_x**2 + local_bev_y**2)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-local_bev_y, local_bev_x)/ configx.h_res_rad
    y_img = np.arctan2(y_road, xy_euclid)/ configx.v_res_rad


    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / configx.h_res / 2  # Theoretical min x value based on sensor specs
    x_img = x_img - x_min              # Shift
    # x_max = int(360.0 / configx.h_res)       # Theoretical max x value after shifting

    y_min = configx.v_fov[0] / configx.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min              # Shift
    y_max = int(configx.v_fov_total / configx.v_res) # Theoretical max x value after shifting

    # y_max = int(y_max + configx.y_fudge)            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.
    y_img = -1 * (y_img - y_max) #di flip

    #batasan
    x_img = np.clip(int(x_img), 0, configx.front_w-1)#constrain
    y_img = np.clip(int(y_img), 0, configx.front_h-1)#constrain

    return x_img, y_img



def bias_slope(angle_x, bias_a, bias_b, angle_a, angle_b):
    bias_x = (((angle_x-angle_a)/(angle_b-angle_a)) * (bias_b-bias_a)) + bias_a
    return bias_x

def bearing_biasing(in_angle, bearing_bias):
    if 0 <= in_angle < 50:
        bias_x = bearing_bias[0]
    elif 50 <= in_angle < 70:
        bias_x = bias_slope(in_angle, bearing_bias[0], bearing_bias[1], 50, 70)
    elif 70 <= in_angle < 110:
        bias_x = bearing_bias[1]
    elif 110 <= in_angle < 130:
        bias_x = bias_slope(in_angle, bearing_bias[1], bearing_bias[2], 110, 130)
    elif 130 <= in_angle < 170:
        bias_x = bearing_bias[2]
    elif 170 <= in_angle <= 180:
        bias_x = bias_slope(in_angle, bearing_bias[2], (bearing_bias[2]+bearing_bias[3])/2, 170, 180)
    elif -180 <= in_angle < -170:
        bias_x = bias_slope(in_angle, (bearing_bias[2]+bearing_bias[3])/2, bearing_bias[3], -180, -170)
    elif -170 <= in_angle < -130:
        bias_x = bearing_bias[3]
    elif -130 <= in_angle < -110:
        bias_x = bias_slope(in_angle, bearing_bias[3], bearing_bias[4], -130, -110)
    elif -110 <= in_angle < -70:
        bias_x = bearing_bias[4]
    elif -70 <= in_angle < -50:
        bias_x = bias_slope(in_angle, bearing_bias[4], bearing_bias[5], -70, -50)
    elif -50 <= in_angle < 0:
        bias_x = bearing_bias[5]
    else:
        bias_x = 0
    biased_angle = in_angle+bias_x

    #normalkan
    if biased_angle > 180: #buat jadi -180 ke 0
        bearing_veh_deg = biased_angle - 360
    elif biased_angle < -180: #buat jadi 180 ke 0
        bearing_veh_deg = biased_angle + 360
    else:
        bearing_veh_deg = biased_angle

    return bearing_veh_deg



#khusus buat inference, bs=1, no depth
def gen_bev_front_seg_infer(self, ptx, pty, ptz, ptseg):
    #flatten all
    ptx = ptx.ravel()
    pty = pty.ravel()
    ptz = ptz.ravel() - self.cover_area_f[0]
    d_lidar = torch.sqrt(ptx**2 + ptz**2) #jarak relatif #+ pty**2 
    ptseg = ptseg.ravel()
    
    #normalize ke frame untuk BEV projection
    frame_data_x = torch.round((ptx+self.cover_area_lr) * (self.bev_w-1) / (2*self.cover_area_lr))
    frame_data_z = torch.round((ptz * (1-self.bev_h) / (self.cover_area_f[1]-self.cover_area_f[0])) + (self.bev_h-1))

    #BEV SEG
    #cari index interest
    boolx = torch.logical_and(frame_data_x <= self.bev_w-1, frame_data_x >= 0)
    bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_z <= self.bev_h-1, frame_data_z >= 0))
    idx = bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    coorx = torch.stack([ptseg, frame_data_z, frame_data_x])
    coor_clsn = torch.unique(coorx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    bev_seg = torch.zeros((1, self.n_class, self.bev_h, self.bev_w), dtype=self.dtype, device=self.gpu_device)
    bev_seg[0, coor_clsn[0], coor_clsn[1], coor_clsn[2]] = 1.0 #format axis dari NCHW


    #FRONT SEG
    #baca: https://github.com/collector-m/lidar_projection/blob/master/show.py
    # Distance relative to origin when looked from top
    # d_lidar = torch.sqrt(ptx**2 + ptz**2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)


    # PROJECT INTO IMAGE COORDINATES
    x_img = torch.atan2(-ptz, ptx)/ self.h_res_rad
    y_img = torch.atan2(pty, d_lidar)/ self.v_res_rad


    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / self.h_res / 2  # Theoretical min x value based on sensor specs
    x_img = x_img - x_min              # Shift
    # x_max = int(360.0 / self.h_res)       # Theoretical max x value after shifting

    y_min = self.v_fov[0] / self.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min             # Shift
    y_max = int(self.v_fov_total / self.v_res) # Theoretical max x value after shifting

    # y_max = int(y_max + self.y_fudge)            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.
    y_img = -1 * (y_img - y_max) #di flip

    #normalize ke frame untuk FRONT projection
    #baca https://towardsdatascience.com/spherical-projection-for-point-clouds-56a2fc258e6c
    # ptR = torch.sqrt(torch.pow(ptx,2) + torch.pow(ptx,2) + torch.pow(ptx,2))
    # pt_pitch = torch.asin(ptz/ptR)
    # pt_yaw = torch.atan2(pty,ptx)
    # frame_data_pitch = (self.front_h-1) * (1-(pt_pitch-self.fov_down)/self.fov)
    # frame_data_yaw = (self.front_w-1) * (0.5*((pt_yaw/np.pi)+1))
    # frame_data_y = torch.round(((pty-self.cover_area_up[0]) * (1-self.front_h) / (self.cover_area_up[1]-self.cover_area_up[0])) + (self.front_h-1))

    #cari index interest
    # boolxz = torch.logical_and(boolx, ptz >= self.cover_area_f[0]) #ptz >= self.cover_area_f[0] berarti point2 yang berada didepan vehicle saja
    # bool_all = torch.logical_and(boolxz, torch.logical_and(frame_data_y <= self.front_h-1, frame_data_y >= 0))
    # boolx = torch.logical_and(frame_data_yaw <= self.front_w-1, frame_data_yaw >= 0)
    # bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_pitch <= self.front_h-1, frame_data_pitch >= 0))
    boolx = torch.logical_and(x_img <= self.front_w-1, x_img >= 0)
    bool_all = torch.logical_and(boolx, torch.logical_and(y_img <= self.front_h-1, y_img >= 0))
    idx = bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    # coorx = torch.stack([ptn, ptseg, frame_data_y, frame_data_x])
    # coorx = torch.stack([ptn, ptseg, frame_data_pitch, frame_data_yaw])
    coorx = torch.stack([ptseg, y_img, x_img])
    coor_clsn = torch.unique(coorx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    # front_seg = torch.zeros((bsize, self.n_class, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
    front_seg = torch.zeros((1, self.n_class, self.front_h, self.front_w), dtype=self.dtype, device=self.gpu_device)
    front_seg[0, coor_clsn[0], coor_clsn[1], coor_clsn[2]] = 1.0 #format axis dari NCHW


    return bev_seg, front_seg





"""
def crop_matrix(image, resize=1, D3=True, crop=[512, 1024]):
    
    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else: #buat matrix 2d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]

    #resize image
    WH_resized = (int(cropped_im.shape[1]/resize), int(cropped_im.shape[0]/resize))
    resized_image = cv2.resize(cropped_im, WH_resized, interpolation=cv2.INTER_NEAREST)

    return resized_image

def swap_RGB2BGR(matrix):
    red = matrix[:,:,0].copy()
    blue = matrix[:,:,2].copy()
    matrix[:,:,0] = blue
    matrix[:,:,2] = red
    return matrix




def cls2one_hot(ss_gt, n_class):
    #inputnya adalah HWC baca cv2 secara biasanya, ambil salah satu channel saja
    ss_gt = np.transpose(ss_gt, (2,0,1)) #GANTI CHANNEL FIRST
    ss_gt = ss_gt[:1,:,:].reshape(ss_gt.shape[1], ss_gt.shape[2])
    result = (np.arange(n_class) == ss_gt[...,None]).astype(int) # jumlah class di cityscape pallete
    result = np.transpose(result, (2, 0, 1))   # (H, W, C) --> (C, H, W)
    # np.save("00009_ss.npy", result) #SUDAH BENAR!
    # print(result)
    # print(result.shape)
    return result

"""