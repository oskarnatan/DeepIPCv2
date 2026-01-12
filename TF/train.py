import pandas as pd
import os
import cv2
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import shutil
from model_lf import late_fusion
from model_gf import geometric_fusion
from model_tf import transfuser
from data import WHILL_Data#, gen_bev_front_seg_dep#, custom_collate
from config import GlobalConfig
from torch.utils.tensorboard import SummaryWriter
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update loss
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


#FUNGSI TRAINING
def train(data_loader, model, config, writer, cur_epoch, optimizer):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'wp_loss': AverageMeter()}
    
    #masuk ke mode training, pytorch
    model.train()

    #visualisasi progress training dengan tqdm
    prog_bar = tqdm(total=len(data_loader))

    #training....
    total_batch = len(data_loader)
    batch_ke = 0
    for data in data_loader:
        cur_step = cur_epoch*total_batch + batch_ke

        #pindah ke torch gpu device dulu
        rgbs = []
        pt_cld_hists = []
        for i in range(0, config.seq_len):
            rgbs.append(torch.tensor(data['rgbs'][i]).to(config.gpu_device, dtype=config.dtype))
            pt_cld_hists.append(torch.tensor(data['pt_cld_hists'][i]).to(config.gpu_device, dtype=config.dtype))
        gt_velocity = torch.stack(data['lr_velo'], dim=1).to(config.gpu_device, dtype=config.dtype)
        rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
        rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
        gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

        #forward pass
        if config.model == 'transfuser':
            pred_wp = model(rgbs, pt_cld_hists, rp1, rp2, gt_velocity)
        else:
            pred_wp = model(rgbs, pt_cld_hists, rp1, rp2)

        #compute loss
        loss_wp = F.l1_loss(pred_wp, gt_waypoints)
        total_loss = loss_wp

        #backpro, kalkulasi gradient, dan optimasi
        optimizer.zero_grad()
        total_loss.backward() #ga usah retain graph
        optimizer.step() #dan update bobot2 pada network model

        #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
        score['total_loss'].update(total_loss.item())
        score['wp_loss'].update(loss_wp.item())

        #update visualisasi progress bar
        postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
                            ('t_wp_l', score['wp_loss'].avg)])
        
        #tambahkan ke summary writer
        writer.add_scalar('t_total_l', total_loss.item(), cur_step)
        writer.add_scalar('t_wp_l', loss_wp.item(), cur_step)

        prog_bar.set_postfix(postfix)
        prog_bar.update(1)
        batch_ke += 1
    prog_bar.close()    

    #return value
    return postfix


#FUNGSI VALIDATION
def validate(data_loader, model, config, writer, cur_epoch):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'wp_loss': AverageMeter()}
            
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        total_batch = len(data_loader)
        batch_ke = 0
        for data in data_loader:
            cur_step = cur_epoch*total_batch + batch_ke

            #pindah ke torch gpu device dulu
            rgbs = []
            pt_cld_hists = []
            for i in range(0, config.seq_len):
                rgbs.append(torch.tensor(data['rgbs'][i]).to(config.gpu_device, dtype=config.dtype))
                pt_cld_hists.append(torch.tensor(data['pt_cld_hists'][i]).to(config.gpu_device, dtype=config.dtype))
            gt_velocity = torch.stack(data['lr_velo'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

            #forward pass
            if config.model == 'transfuser':
                pred_wp = model(rgbs, pt_cld_hists, rp1, rp2, gt_velocity)
            else:
                pred_wp = model(rgbs, pt_cld_hists, rp1, rp2)

            #compute loss
            loss_wp = F.l1_loss(pred_wp, gt_waypoints)
            total_loss = loss_wp

            #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
            score['total_loss'].update(total_loss.item())
            score['wp_loss'].update(loss_wp.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
                                ('v_wp_l', score['wp_loss'].avg)])
            
            #tambahkan ke summary writer
            writer.add_scalar('v_total_l', total_loss.item(), cur_step)
            writer.add_scalar('v_wp_l', loss_wp.item(), cur_step)

            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
            batch_ke += 1
        prog_bar.close()    

    #return value
    return postfix


#MAIN FUNCTION
def main():
    # Load config
    config = GlobalConfig()
    
    #SET GPU YANG AKTIF
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

    #IMPORT MODEL UNTUK DITRAIN
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    if config.model == 'late_fusion':
        model = late_fusion(config)
    elif config.model == 'geometric_fusion':
        model = geometric_fusion(config)
    else:
        model = transfuser(config)
    model.to(config.gpu_device, dtype=config.dtype)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    #KONFIGURASI OPTIMIZER
    # optima = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optima = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    #BUAT DATA BATCH
    train_set = WHILL_Data(data_root=config.train_dir, conditions=config.train_conditions, config=config)
    val_set = WHILL_Data(data_root=config.val_dir, conditions=config.val_conditions, config=config)
    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True) #, collate_fn=custom_collate
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True) #, collate_fn=custom_collate
    # print(len(dataloader_train))
    
    #cek retrain atau tidak
    if not os.path.exists(config.logdir+"/trainval_log.csv"):
        print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)
        #set nilai awal
        curr_ep = 0
        lowest_score = float('inf')
        stop_count = config.init_stop_counter
    else:
        print('Continue training!!!!!!!!!!!!!!!!')
        print('Loading checkpoint from ' + config.logdir)
        #baca log history training sebelumnya
        log_trainval = pd.read_csv(config.logdir+"/trainval_log.csv")
        # replace variable2 ini
        # print(log_trainval['epoch'][-1:])
        curr_ep = int(log_trainval['epoch'][-1:]) + 1
        lowest_score = float(np.min(log_trainval['val_loss']))
        stop_count = int(log_trainval['stop_counter'][-1:])
        # Load checkpoint
        model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
        optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))
        #update direktori dan buat tempat penyimpanan baru
        config.logdir += "/retrain"
        os.makedirs(config.logdir, exist_ok=True)
        print('Created new retrain dir:', config.logdir)
    
    #copykan config file
    shutil.copyfile('config.py', config.logdir+'/config.py')

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
            ('epoch', []),
            ('best_model', []),
            ('val_loss', []),
            ('val_wp_loss', []),
            ('train_loss', []), 
            ('train_wp_loss', []),
            ('lrate', []),
            ('stop_counter', []), 
            ('elapsed_time', []),
        ])
    writer = SummaryWriter(log_dir=config.logdir)
    
    #proses iterasi tiap epoch
    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        #cetak lr 
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        #training validation
        start_time = time.time() #waktu mulai
        train_log = train(dataloader_train, model, config, writer, epoch, optima)
        val_log = validate(dataloader_val, model, config, writer, epoch)
        #update learning rate untuk training process
        scheduler.step(val_log['v_total_l']) #parameter acuan reduce LR adalah val_total_metric
        elapsed_time = time.time() - start_time #hitung elapsedtime

        #simpan history training ke file csv
        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['t_total_l'])
        log['val_loss'].append(val_log['v_total_l'])
        log['train_wp_loss'].append(train_log['t_wp_l'])
        log['val_wp_loss'].append(val_log['v_wp_l'])
        log['elapsed_time'].append(elapsed_time)
        print('| t_total_l: %.4f | t_wp_l: %.4f |' % (train_log['t_total_l'], train_log['t_wp_l']))
        print('| v_total_l: %.4f | v_wp_l: %.4f |' % (val_log['v_total_l'], val_log['v_wp_l']))
        print('elapsed time: %.4f sec' % (elapsed_time))
        
        #save recent model dan optimizernya
        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))

        #save model best only
        if val_log['v_total_l'] < lowest_score:
            print("v_total_l: %.4f < lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model terbaik disave!")
            torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
            torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
            # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'best_optim_lw.pth'))
            #v_total_l sekarang menjadi lowest_score
            lowest_score = val_log['v_total_l']
            #reset stop counter
            stop_count = config.init_stop_counter
            print("stop counter direset ke: ", stop_count)
            #catat sebagai best model
            log['best_model'].append("BEST")
        else:
            print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model tidak disave!")
            stop_count -= 1
            print("stop counter : ", stop_count)
            log['best_model'].append("")

        #update stop counter
        log['stop_counter'].append(stop_count)
        #paste ke csv file
        pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

        #kosongkan cuda chace
        torch.cuda.empty_cache()
        epoch += 1

        # early stopping jika stop counter sudah mencapai 0 dan early stop true
        if stop_count==0:
            print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
            break #loop
        

#RUN PROGRAM
if __name__ == "__main__":
    main()


