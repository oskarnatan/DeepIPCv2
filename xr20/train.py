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
from model import xr20
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


#fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config):
    #detach dulu paramsnya dari torch, pindah ke CPU
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])
    #fungsi renormalize untuk algoritma 1 di papaer gradnorm
    coef = np.array(config.loss_weights).sum()/lws.sum()
    new_lws = [coef*lwx for lwx in lws]
    #buat torch float tensor lagi dan masukkan ke cuda memory
    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws

#FUNGSI TRAINING
def train(data_loader, model, config, writer, cur_epoch, optimizer, params_lw, optimizer_lw):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'wp_loss': AverageMeter(),
            'str_loss': AverageMeter(),
            'thr_loss': AverageMeter()}
    
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
        bev_segs = []
        bev_deps = []
        front_segs = []
        front_deps = []
        for i in range(0, config.seq_len):
            bev_segs.append(torch.tensor(data['bev_segs'][i]).to(config.gpu_device, dtype=config.dtype))
            bev_deps.append(torch.tensor(data['bev_deps'][i]).to(config.gpu_device, dtype=config.dtype))
            front_segs.append(torch.tensor(data['front_segs'][i]).to(config.gpu_device, dtype=config.dtype))
            front_deps.append(torch.tensor(data['front_deps'][i]).to(config.gpu_device, dtype=config.dtype))

        rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
        rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
        gt_velocity = torch.stack(data['lr_velo'], dim=1).to(config.gpu_device, dtype=config.dtype)
        gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)
        gt_steering = data['steering'].to(config.gpu_device, dtype=config.dtype)
        gt_throttle = data['throttle'].to(config.gpu_device, dtype=config.dtype)

        #forward pass
        pred_wp, pred_steering, pred_throttle = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity, data['cmd'])
        # check_gt_seg(config, sdcs[-1])

        #compute loss
        loss_wp = F.l1_loss(pred_wp, gt_waypoints)
        loss_str = F.l1_loss(pred_steering, gt_steering)
        loss_thr = F.l1_loss(pred_throttle, gt_throttle)
        total_loss = params_lw[0]*loss_wp + params_lw[1]*loss_str + params_lw[2]*loss_thr

        #backpro, kalkulasi gradient, dan optimasi
        optimizer.zero_grad()

        if batch_ke == 0: #batch pertama, hitung loss awal
            total_loss.backward() #ga usah retain graph
            #ambil loss pertama
            loss_wp_0 = torch.clone(loss_wp)
            loss_str_0 = torch.clone(loss_str)
            loss_thr_0 = torch.clone(loss_thr)

        elif 0 < batch_ke < total_batch-1:
            total_loss.backward() #ga usah retain graph

        elif batch_ke == total_batch-1: #berarti batch terakhir, compute update loss weights
            if config.MGN:
                optimizer_lw.zero_grad()
                total_loss.backward(retain_graph=True) #backpro, hitung gradient, retain graph karena graphnya masih dipakai perhitungan
                #ambil nilai gradient dari layer pertama pada masing2 task-specified decoder dan komputasi gradient dari output layer sampai ke bottle neck saja
                params = list(filter(lambda p: p.requires_grad, model.parameters()))
                G0R = torch.autograd.grad(loss_wp, params[config.bottleneck], retain_graph=True, create_graph=True)
                G0 = torch.norm(G0R[0], keepdim=True)
                G1R = torch.autograd.grad(loss_str, params[config.bottleneck], retain_graph=True, create_graph=True)
                G1 = torch.norm(G1R[0], keepdim=True)
                G2R = torch.autograd.grad(loss_thr, params[config.bottleneck], retain_graph=True, create_graph=True)
                G2 = torch.norm(G2R[0], keepdim=True)
                #dan rata2
                G_avg = (G0+G1+G2) / len(config.loss_weights)

                #hitung relative lossnya
                loss_wp_hat = loss_wp / loss_wp_0
                loss_str_hat = loss_str / loss_str_0
                loss_thr_hat = loss_thr / loss_thr_0
                #dan rata2
                loss_hat_avg = (loss_wp_hat + loss_str_hat + loss_thr_hat) / len(config.loss_weights)

                #hitung r_i_(t) relative inverse training rate untuk setiap task 
                inv_rate_wp = loss_wp_hat / loss_hat_avg
                inv_rate_str = loss_str_hat / loss_hat_avg
                inv_rate_thr = loss_thr_hat / loss_hat_avg

                #hitung constant target grad
                C0 = (G_avg*inv_rate_wp).detach()**config.lw_alpha
                C1 = (G_avg*inv_rate_str).detach()**config.lw_alpha
                C2 = (G_avg*inv_rate_thr).detach()**config.lw_alpha

                #HITUNG TOTAL LGRAD
                Lgrad = F.l1_loss(G0, C0) + F.l1_loss(G1, C1) + F.l1_loss(G2, C2)

                #hitung gradient loss sesuai Eq. 2 di GradNorm paper
                # optimizer_lw.zero_grad()
                Lgrad.backward()
                #update loss weights
                optimizer_lw.step() 

                #ambil lgrad untuk disimpan nantinya
                lgrad = Lgrad.item()
                new_param_lw = optimizer_lw.param_groups[0]['params']
                # print(new_param_lw)
            else:
                total_loss.backward()
                lgrad = 0
                new_param_lw = 1
            
        optimizer.step() #dan update bobot2 pada network model

        #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
        score['total_loss'].update(total_loss.item())
        score['wp_loss'].update(loss_wp.item())
        score['str_loss'].update(loss_str.item())
        score['thr_loss'].update(loss_thr.item())

        #update visualisasi progress bar
        postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
                            ('t_wp_l', score['wp_loss'].avg),
                            ('t_str_l', score['str_loss'].avg),
                            ('t_thr_l', score['thr_loss'].avg)])
        
        #tambahkan ke summary writer
        writer.add_scalar('t_total_l', total_loss.item(), cur_step)
        writer.add_scalar('t_wp_l', loss_wp.item(), cur_step)
        writer.add_scalar('t_str_l', loss_str.item(), cur_step)
        writer.add_scalar('t_thr_l', loss_thr.item(), cur_step)

        prog_bar.set_postfix(postfix)
        prog_bar.update(1)
        batch_ke += 1
    prog_bar.close()    

    #return value
    return postfix, new_param_lw, lgrad


#FUNGSI VALIDATION
def validate(data_loader, model, config, writer, cur_epoch):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'wp_loss': AverageMeter(),
            'str_loss': AverageMeter(),
            'thr_loss': AverageMeter()}
            
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
            bev_segs = []
            bev_deps = []
            front_segs = []
            front_deps = []
            for i in range(0, config.seq_len):
                bev_segs.append(torch.tensor(data['bev_segs'][i]).to(config.gpu_device, dtype=config.dtype))
                bev_deps.append(torch.tensor(data['bev_deps'][i]).to(config.gpu_device, dtype=config.dtype))
                front_segs.append(torch.tensor(data['front_segs'][i]).to(config.gpu_device, dtype=config.dtype))
                front_deps.append(torch.tensor(data['front_deps'][i]).to(config.gpu_device, dtype=config.dtype))

            rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_velocity = torch.stack(data['lr_velo'], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_steering = data['steering'].to(config.gpu_device, dtype=config.dtype)
            gt_throttle = data['throttle'].to(config.gpu_device, dtype=config.dtype)

            #forward pass
            pred_wp, pred_steering, pred_throttle = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity, data['cmd'])

            #compute loss
            loss_wp = F.l1_loss(pred_wp, gt_waypoints)
            loss_str = F.l1_loss(pred_steering, gt_steering)
            loss_thr = F.l1_loss(pred_throttle, gt_throttle)
            total_loss = loss_wp + loss_str + loss_thr

            #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
            score['total_loss'].update(total_loss.item())
            score['wp_loss'].update(loss_wp.item())
            score['str_loss'].update(loss_str.item())
            score['thr_loss'].update(loss_thr.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
                                ('v_wp_l', score['wp_loss'].avg),
                                ('v_str_l', score['str_loss'].avg),
                                ('v_thr_l', score['thr_loss'].avg)])
            
            #tambahkan ke summary writer
            writer.add_scalar('v_total_l', total_loss.item(), cur_step)
            writer.add_scalar('v_wp_l', loss_wp.item(), cur_step)
            writer.add_scalar('v_str_l', loss_str.item(), cur_step)
            writer.add_scalar('v_thr_l', loss_thr.item(), cur_step)

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
    model = xr20(config).to(config.gpu_device, dtype=config.dtype)
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
    # print(len(train_set))
    if len(train_set)%config.batch_size == 1:
        drop_last = True #supaya ga mengacaukan MGN #drop last perlu untuk MGN
    else: #selain 1 bisa
        drop_last = False
    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last) #, collate_fn=custom_collate
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True) #, collate_fn=custom_collate
    # print(len(dataloader_train))
    
    #cek retrain atau tidak
    if not os.path.exists(config.logdir+"/trainval_log.csv"):
        print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)
        #optimizer lw
        params_lw = [torch.cuda.FloatTensor([config.loss_weights[i]]).clone().detach().requires_grad_(True) for i in range(len(config.loss_weights))]
        optima_lw = optim.SGD(params_lw, lr=config.lr)
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

        #set optima lw baru
        latest_lw = [float(log_trainval['lw_wp'][-1:]), float(log_trainval['lw_str'][-1:]), float(log_trainval['lw_thr'][-1:])]
        params_lw = [torch.cuda.FloatTensor([latest_lw[i]]).clone().detach().requires_grad_(True) for i in range(len(latest_lw))]
        optima_lw = optim.SGD(params_lw, lr=float(log_trainval['lrate'][-1:]))
        # optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] # lr disamakan
        # optima_lw.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim_lw.pth')))
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
            ('val_str_loss', []),
            ('val_thr_loss', []),
            ('train_loss', []), 
            ('train_wp_loss', []),
            ('train_str_loss', []),
            ('train_thr_loss', []),
            ('lrate', []),
            ('stop_counter', []), 
            ('lgrad_loss', []),
            ('lw_wp', []),
            ('lw_str', []),
            ('lw_thr', []),
            ('elapsed_time', []),
        ])
    writer = SummaryWriter(log_dir=config.logdir)
    
    #proses iterasi tiap epoch
    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        #cetak lr dan lw
        if config.MGN:
            curr_lw = optima_lw.param_groups[0]['params']
            lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
            lws = np.array([lw[i][0] for i in range(len(lw))])
            print("current loss weights: ", lws)    
        else:
            curr_lw = config.loss_weights
            lws = config.loss_weights
            print("current loss weights: ", config.loss_weights)
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        #training validation
        start_time = time.time() #waktu mulai
        train_log, new_params_lw, lgrad = train(dataloader_train, model, config, writer, epoch, optima, curr_lw, optima_lw)
        val_log = validate(dataloader_val, model, config, writer, epoch)
        if config.MGN:
            #update params lw yang sudah di renormalisasi ke optima_lw
            optima_lw.param_groups[0]['params'] = renormalize_params_lw(new_params_lw, config) #harus diclone supaya benar2 terpisah
            print("total loss gradient: "+str(lgrad))
        #update learning rate untuk training process
        scheduler.step(val_log['v_total_l']) #parameter acuan reduce LR adalah val_total_metric
        optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] #update lr disamakan
        elapsed_time = time.time() - start_time #hitung elapsedtime

        #simpan history training ke file csv
        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['t_total_l'])
        log['val_loss'].append(val_log['v_total_l'])
        log['train_wp_loss'].append(train_log['t_wp_l'])
        log['val_wp_loss'].append(val_log['v_wp_l'])
        log['train_str_loss'].append(train_log['t_str_l'])
        log['val_str_loss'].append(val_log['v_str_l'])
        log['train_thr_loss'].append(train_log['t_thr_l'])
        log['val_thr_loss'].append(val_log['v_thr_l'])
        log['lgrad_loss'].append(lgrad)
        log['lw_wp'].append(lws[0])
        log['lw_str'].append(lws[1])
        log['lw_thr'].append(lws[2])
        log['elapsed_time'].append(elapsed_time)
        print('| t_total_l: %.4f | t_wp_l: %.4f | t_str_l: %.4f | t_thr_l: %.4f |' % (train_log['t_total_l'], train_log['t_wp_l'], train_log['t_str_l'], train_log['t_thr_l']))
        print('| v_total_l: %.4f | v_wp_l: %.4f | v_str_l: %.4f | v_thr_l: %.4f |' % (val_log['v_total_l'], val_log['v_wp_l'], val_log['v_str_l'], val_log['v_thr_l']))
        print('elapsed time: %.4f sec' % (elapsed_time))
        
        #save recent model dan optimizernya
        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))
        # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'recent_optim_lw.pth'))

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


