from glob import glob

env = DefaultEnvironment()
env.Append(
    CXXFLAGS="-O2 -Wall -std=c++11 -fdiagnostics-color=auto",
    LINKFLAGS="-Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,--as-needed",
    CPPPATH=['/usr/local/include'],
    LIBPATH=['/usr/local/lib', '/usr/local/lib64'],
)

env.Append(
    CXXFLAGS=['-DA64 -fopenmp'],
    CPPPATH=['detectors/mser/LL/', 'detectors/mser/utls/'],
    LIBS=['gomp', 'lapack']+ ['opencv_'+i for i in ["core", "highgui", "imgproc", "flann", "video", "features2d", "calib3d"]]
)

srcs = glob('*.cpp')
srcs.remove('mods.cpp')
srcs.remove('io_mods.cpp')
srcs += glob('USAC/src/*/*.cpp') + glob('detectors/*/*.c*') + glob('detectors/*/*/*.c*')
SharedLibrary('mods', srcs)
StaticLibrary('mods_s', srcs)
Program('run_mods', ['mods.cpp', 'io_mods.cpp', 'libmods_s.a'] + glob('inih/*.c') + glob('inih/cpp/INIReader.cpp'))
