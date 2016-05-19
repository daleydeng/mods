#OMP_NUM_THREADS=1 for debug
n1=img_a.jpg
n2=img_b.jpg
OMP_NUM_THREADS=1 ../run_mods $n1 $n2 imgOut1.png imgOut2.png keys1.txt keys2.txt matchings.txt log.txt 0 H ../config_iter_wxbs.ini ../iters_wxbs.ini
