TRAIN-VAL-TEST:
1. Open the config.py file and adjust the dataset directory and other parameters.
2. python3 train.py (you will obtain a .pth file of the model)
3. python3 test.py (you will obtain the score for the offline test)


ARCHITECTURE:\
<img src="docs/xr20_arch.png" width="600">
<img src="docs/encoder_fusion_block.png" width="600">


AUTO DRIVING:\
<img src="docs/xr20_seg_seq1_2022-10-28_route21.gif" width="600">


Local route points to high-level navigational commands:\
<img src="docs/rp2cmd.png" width="600">