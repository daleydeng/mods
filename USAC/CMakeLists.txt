CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(usac)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

include_directories(.)
include_directories(./src)
include_directories(./src/config)
include_directories(./src/estimators)
include_directories(./src/utils)

set(usac_srcs
src/config/ConfigParams.cpp
src/config/ConfigParamsHomog.cpp
src/config/ConfigParamsFundmatrix.cpp
src/utils/FundmatrixFunctions.cpp
src/utils/HomographyFunctions.cpp
src/utils/MathFunctions.cpp)

add_library(usac ${usac_srcs})
target_link_libraries(usac lapack)