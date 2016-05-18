MODS: Image Matching with On-Demand Synthesis.

Compilation. 
MODS depends on OpenCV version 2.4.0 - 2.4.10 and LAPACK

Compile by:
cd build
cmake ..
make

Example of use:
Linux (mods_example.sh runs this example):
./mods ../dataset/1/cat.jpg ../dataset/2/cat.jpg imgOut1.png imgOut2.png k1.txt k2.txt m.txt log.txt 0 H config_iter_cviu.ini iters_cviu.ini
- dataset/1/cat.jpg, dataset/2/cat.jpg: input images;
- imgOut1.png, imgOut2.png: output images. The detected matchings are represented by green and blue dots or ellipses
- k1.txt k2.txt: affine regions and their descriptors of the two images.
- m.txt: coordinates of matched points x1 y1 x2 y2
- log.txt - log-file for graphs
- write log file only [0/1]. If 1, no other files will be generated.
- geometry verification type [H/F]

Configurations:

config_iter_cviu.ini, iters_cviu.ini - version, created to hangle extreme view changes. 
Described in   
"MODS: Fast and Robust Method for Two-View Matching" by Dmytro Mishkin, Jiri Matas, Michal Perdoch.
http://arxiv.org/abs/1503.02619.

config_iter_wxbs.ini, iters_wxbs.ini - version, described in . 
"WxBS: Wide Baseline Stereo Generalizations" by Dmytro Mishkin, Jiri Matas, Michal Perdoch, Karel Lenc.
http://arxiv.org/abs/1504.06603
It handles extreme appearance and geometrical changes. A bit slower than previous, but much more powerful.
If use, please cite corresponding papers.

Configuration files are very flexible, we encourage you to try your own view synthesis-detector-descriptor combinations

!Note:
Due to licence issues, we replaces internal RANSAC version used in MODS binaries IVCNZ-2013 (still available at http://cmp.felk.cvut.cz/wbs ) 
with free available USAC, which however gives inferior results in high number of outliers.
Therefore results of this version are not directly comparable to the ones stated in paper, but are close enough.
