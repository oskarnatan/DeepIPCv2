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

        data['rgbs'] = []
        data['pt_cld_hists'] = []
        # data['segs'] = []
        # data['ptxs'] = []
        # data['ptys'] = []
        # data['ptzs'] = []
        # data['ptss'] = []
        # data['velodyne'] = []
        # data['polarnet'] = []
        # data['bev_segs'] = []
        # data['bev_deps'] = []
        # data['front_segs'] = []
        # data['front_deps'] = []
        seq_rgbs = self.rgb[index]
        # seq_segs = self.seg[index]
        # seq_pt_clouds = self.pt_cloud[index]
        seq_pt_cloud_pcds = self.pt_cld_pcd[index]
        # seq_pt_cloud_segs = self.pt_cld_seg[index]
        seq_loc_xs = self.loc_x[index]
        seq_loc_ys = self.loc_y[index]
        seq_loc_headings = self.loc_heading[index]

        for i in range(0, self.seq_len):
            #RGB
            data['rgbs'].append(np.array(resize_img(cv2.imread(seq_rgbs[i]), resize_w=self.config.front_w , resize_h=self.config.front_h)).transpose(2,0,1))
            #lidar
            lid_pc = pypcd.PointCloud.from_path(seq_pt_cloud_pcds[i])
            lid_x = lid_pc.pc_data['x']
            lid_y = lid_pc.pc_data['y']
            lid_z = lid_pc.pc_data['z']
            # lid_intensity = lid_pc.pc_data['intensity']
            in_velodyne = np.zeros(lid_x.shape[0] + lid_y.shape[0] + lid_z.shape[0], dtype=np.float32) # + lid_intensity.shape[0]
            #INPUT POINT CLOUD YANG DIGENERATE DARI HDL32E
            in_velodyne[0::3] = -1*(lid_y - self.config.cover_area_f[0]) #lid_x
            in_velodyne[1::3] = lid_x #lid_y
            in_velodyne[2::3] = lid_z
            # in_velodyne[3::4] = lid_intensity
            full_lidar = in_velodyne.astype('float32').reshape((-1, 3)) #formatnya XYZI lalu diambil XYZnya saja #
            #baca data.py nya LF atau GF atau TF
            # convert coordinate frame of point cloud
            # full_lidar[:,1] *= -1 # inverts x, y
            lidar_processed = lidar_to_histogram_features(full_lidar, self.config)
            data['pt_cld_hists'].append(lidar_processed.transpose(2,0,1))
            
            """
            #untuk geometric fusion
            bev_points, cam_points = lidar_bev_cam_correspondences(full_lidar, crop=self.input_resolution)
            data['bev_points'].append(bev_points)
            data['cam_points'].append(cam_points)
            """

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

        #vehicular controls dan velocity jadikan satu LR
        data['steering'] = self.steering[index]
        data['throttle'] = self.throttle[index]
        data['lr_velo'] = tuple(np.array([self.velocity_l[index], self.velocity_r[index]]))

        #metadata buat testing nantinya
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data





def lidar_to_histogram_features(lidar, configx):
    def splat_points(point_cloud):
        #128 x 256 grid
        x_meters_max = int(configx.cover_area_lr)
        y_meters_max = int((configx.cover_area_f[1] - configx.cover_area_f[0]) / 2)
        pixels_per_meter = int(configx.bev_w/configx.cover_area_lr)
        hist_max_per_pixel = 5
        xbins = np.linspace(-x_meters_max, x_meters_max+1, x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(ybins, xbins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=0] #di bawah atau sama dengan garis 0 horizon lidar
    above = lidar[lidar[...,2]>0] #di atas garis 0 horizon lidar
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = features.astype(np.float32) #np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


def resize_img(image, resize_w=256, resize_h=128):
    resized_image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    return resized_image



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




"""
#Convert LiDAR point cloud to camera co-ordinates

def lidar_bev_cam_correspondences(world, crop=256):

    pixels_per_world = 8
    w = 400
    h = 300
    fov = 100
    F = w / (2 * np.tan(fov * np.pi / 360))
    fy = F
    fx = 1.1 * F
    cam_height = 2.3

    # get valid points in 32x32 grid
    world[:,1] *= -1
    lidar = world[abs(world[:,0])<16] # 16m to the sides
    lidar = lidar[lidar[:,1]<32] # 32m to the front
    lidar = lidar[lidar[:,1]>0] # 0m to the back

    # convert to cam space
    z = lidar[..., 1]
    x = (fx * lidar[..., 0]) / z + w / 2
    y = (fy * cam_height) / z + h / 2
    result = np.stack([x, y], 1)
    result[:,0] = np.clip(result[..., 0], 0, w-1)
    result[:,1] = np.clip(result[..., 1], 0, h-1)

    start_x = w // 2 - crop // 2
    start_y = h // 2 - crop // 2
    end_x = start_x + crop
    end_y = start_y + crop

    valid_lidar_points = []
    valid_bev_points = []
    valid_cam_points = []
    for i in range(lidar.shape[0]):
        if result[i][0] >= start_x and result[i][0] < end_x and result[i][1] >= start_y and result[i][1] < end_y:
            result[i][0] -= start_x
            result[i][1] -= start_y
            valid_lidar_points.append(lidar[i])
            valid_cam_points.append([int(result[i][0]), int(result[i][1])])
            bev_x = min(int((lidar[i][0] + 16) * pixels_per_world), crop-1)
            bev_y = min(int(lidar[i][1] * pixels_per_world), crop-1)
            valid_bev_points.append([bev_x, bev_y])

    valid_lidar_points = np.array(valid_lidar_points)
    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points, 8, 32)

    return bev_points, cam_points
"""

