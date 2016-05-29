from glob import glob

env = DefaultEnvironment(
    CCFLAGS="-O2 -Wall -std=c++11 -fdiagnostics-color=auto",
    LINKFLAGS="-Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,--as-needed",
    CPPPATH=['/usr/local/include'],
    LIBPATH=['/usr/local/lib', '/usr/local/lib64'],
)

env.Append(
    CXXFLAGS='-DA64 -fopenmp',
    CPPPATH=['mser/LL/', 'mser/utls/', '/usr/local/opencv3/include', '.', 'app'],
    LIBPATH=['/usr/local/opencv3/lib'],
    LIBS=['gomp', 'lapack']+ ['opencv_'+i for i in ["core", "highgui", "imgproc", "flann", "video", "features2d", "calib3d", 'imgcodecs']]
)

srcs = glob('*.c*')
srcs += glob('USAC/src/*/*.cpp') + glob('mser/*.c*') + glob('mser/*/*.c*')
SharedLibrary('mods', srcs)
StaticLibrary('mods_s', srcs)
Program('run_mods', ['app/mods.cpp', 'app/io_mods.cpp', 'libmods_s.a'] + glob('app/inih/*.c') + glob('app/inih/cpp/INIReader.cpp'))
