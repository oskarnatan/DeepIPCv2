import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
# import cv2
from torch import torch
import yaml

from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model_lf import late_fusion
from model_gf import geometric_fusion
from model_tf import transfuser
from data import WHILL_Data
from log.transfuser_seq1.config import GlobalConfig #pakai config.py yang dicopykan ke log
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update metric
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#FUNGSI test
def test(data_loader, model, config):
    #buat variabel untuk menyimpan kalkulasi metric, dan iou
    score = {'total_metric': AverageMeter(),
            'wp_metric': AverageMeter(),
            'str_metric': AverageMeter(),
            'thr_metric': AverageMeter()}

    #buat object log
    log = OrderedDict([
                    ('batch', []),
                    ('test_metric', []),
                    ('test_wp_metric', []),
                    ('test_str_metric', []),
                    ('test_thr_metric', []),
                    ('model_elapsed_time', []),
                ])
    last_kondisi = '' #flag buat log csv baru
    batch_ke = 1

    #buat save direktori
    save_dir = config.logdir + "/offline_test/" 
    os.makedirs(save_dir, exist_ok=True)
            
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #test        
        for data in data_loader:
            #cek condition terakhir
            if data['condition'][-1] != last_kondisi:
                last_kondisi = data['condition'][-1]
                batch_ke = 1

                if len(log['batch']) != 0:
                    #ketika semua sudah selesai, hitung rata2 performa pada log
                    log['batch'].append("avg")
                    log['test_metric'].append(np.mean(log['test_metric']))
                    log['test_wp_metric'].append(np.mean(log['test_wp_metric']))
                    log['test_str_metric'].append(np.mean(log['test_str_metric']))
                    log['test_thr_metric'].append(np.mean(log['test_thr_metric']))
                    log['model_elapsed_time'].append(np.mean(log['model_elapsed_time']))
                    
                    #ketika semua sudah selesai, hitung VARIANCE performa pada log
                    log['batch'].append("stddev")
                    log['test_metric'].append(np.std(log['test_metric'][:-1]))
                    log['test_wp_metric'].append(np.std(log['test_wp_metric'][:-1]))
                    log['test_str_metric'].append(np.std(log['test_str_metric'][:-1]))
                    log['test_thr_metric'].append(np.std(log['test_thr_metric'][:-1]))
                    log['model_elapsed_time'].append(np.std(log['model_elapsed_time'][:-1]))
                    #paste ke csv file
                    pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)

                #buat save dir log baru
                save_dir_log = save_dir+last_kondisi
                os.makedirs(save_dir_log, exist_ok=True)

                #reset log untuk menyimpan test log baru di CSV
                log = OrderedDict([
                    ('batch', []),
                    ('test_metric', []),
                    ('test_wp_metric', []),
                    ('test_str_metric', []),
                    ('test_thr_metric', []),
                    ('model_elapsed_time', []),
                ])           

            start_time = time.time() #waktu mulai
            #pindah ke torch gpu device dulu
            rgbs = []
            pt_cld_hists = []
            for i in range(0, config.seq_len):
                rgbs.append(torch.tensor(data['rgbs'][i]).to(config.gpu_device, dtype=config.dtype))
                pt_cld_hists.append(torch.tensor(data['pt_cld_hists'][i]).to(config.gpu_device, dtype=config.dtype))
             
            rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_velocity = torch.stack(data['lr_velo'], dim=1).to(config.gpu_device, dtype=config.dtype)
            lr_velo = gt_velocity.cpu().detach().numpy()
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

            #forward pass
            model_start_time = time.time() #waktu mulai
            if config.model == 'transfuser':
                pred_wp = model(rgbs, pt_cld_hists, rp1, rp2, gt_velocity)
            else:
                pred_wp = model(rgbs, pt_cld_hists, rp1, rp2)
            steering, throttle, meta_pred = model.pid_control(pred_wp, lr_velo[-1])
            model_elapsed_time = time.time() - model_start_time #hitung elapsedtime

            #compute metric
            metric_wp = F.l1_loss(pred_wp, gt_waypoints)
            # metric_str = F.l1_loss(psteering, gt_steering)
            # metric_thr = F.l1_loss(pthrottle, gt_throttle)
            metric_str = np.abs(data['steering'].item() - steering)
            metric_thr = np.abs(data['throttle'].item() - throttle)
            total_metric = metric_wp.item() + metric_str + metric_thr

            #hitung rata-rata (avg) metric, dan metric untuk batch-batch yang telah diproses
            score['total_metric'].update(total_metric)#.item())
            score['wp_metric'].update(metric_wp.item())
            score['str_metric'].update(metric_str)#.item())
            score['thr_metric'].update(metric_thr)#.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('te_total_m', score['total_metric'].avg),
                                ('te_wp_m', score['wp_metric'].avg),
                                ('te_str_m', score['str_metric'].avg),
                                ('te_thr_m', score['thr_metric'].avg)])
            
            #simpan history test ke file csv, ambil dari hasil kalkulasi metric langsung, jangan dari averagemeter
            log['batch'].append(batch_ke)
            log['test_metric'].append(total_metric)#.item())
            log['test_wp_metric'].append(metric_wp.item())
            log['test_str_metric'].append(metric_str)#.item())
            log['test_thr_metric'].append(metric_thr)#.item())
            log['model_elapsed_time'].append(model_elapsed_time)
            #paste ke csv file
            pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)

            #save metadata prediksi
            save_dir_meta = save_dir+last_kondisi+'/'+data['route'][-1]+'/pred_meta/'
            os.makedirs(save_dir_meta, exist_ok=True)
            #isikan beberapa data
            meta_pred['rp1_pos_local'] = rp1[0].cpu().detach().numpy().tolist()
            meta_pred['rp2_pos_local'] = rp2[0].cpu().detach().numpy().tolist()
            meta_pred['rp1_pos_global'] = np.array([data['lat_rp1'].item(), data['lon_rp1'].item()]).tolist()
            meta_pred['rp2_pos_global'] = np.array([data['lat_rp2'].item(), data['lon_rp2'].item()]).tolist()
            meta_pred['robot_bearing'] = float(data['bearing_robot'].item())
            meta_pred['robot_pos_global'] = np.array([data['lat_robot'].item(), data['lon_robot'].item()]).tolist()
            meta_pred['model_fps'] = float(1/model_elapsed_time)
            elapsed_time = time.time() - start_time #hitung elapsedtime
            meta_pred['fps'] = float(1/elapsed_time)
            with open(save_dir_meta+data['filename'][-1]+"yml", 'w') as dict_file:
                yaml.dump(meta_pred, dict_file)

            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()
        
        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['test_metric'].append(np.mean(log['test_metric']))
        log['test_wp_metric'].append(np.mean(log['test_wp_metric']))
        log['test_str_metric'].append(np.mean(log['test_str_metric']))
        log['test_thr_metric'].append(np.mean(log['test_thr_metric']))
        log['model_elapsed_time'].append(np.mean(log['model_elapsed_time']))
        
        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['test_metric'].append(np.std(log['test_metric'][:-1]))
        log['test_wp_metric'].append(np.std(log['test_wp_metric'][:-1]))
        log['test_str_metric'].append(np.std(log['test_str_metric'][:-1]))
        log['test_thr_metric'].append(np.std(log['test_thr_metric'][:-1]))
        log['model_elapsed_time'].append(np.std(log['model_elapsed_time'][:-1]))

        #paste ke csv file
        pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)


    #return value
    return log



# Load config
config = GlobalConfig()

#SET GPU YANG AKTIF
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

#IMPORT MODEL dan load bobot
print("IMPORT ARSITEKTUR DL DAN COMPILE")
if config.model == 'late_fusion':
    model = late_fusion(config)
elif config.model == 'geometric_fusion':
    model = geometric_fusion(config)
else:
    model = transfuser(config)
model.to(config.gpu_device, dtype=config.dtype)
model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

#BUAT DATA BATCH
test_set = WHILL_Data(data_root=config.test_dir, conditions=config.test_conditions, config=config)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) #BS selalu 1

#test
test_log = test(dataloader_test, model, config)


#kosongkan cuda chace
torch.cuda.empty_cache()

