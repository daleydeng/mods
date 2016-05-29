#OMP_NUM_THREADS=1 for debug
n1=img_a.jpg
n2=img_b.jpg
cfg=wxbs
mode=H
OMP_NUM_THREADS=4 ../run_mods $n1 $n2 imgOut1.png imgOut2.png keys1.txt keys2.txt matchings.txt log.txt 0 $mode config_iter_$cfg.ini iters_$cfg.ini
