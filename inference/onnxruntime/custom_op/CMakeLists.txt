cmake_minimum_required(VERSION 3.5)
PROJECT(custom_op)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include_directories(${OpenCV_INCLUDE_DIR} ${OnnxRuntime_INCLUDE_DIR})

add_executable(custom_op custom_op.cpp)

# find_package(OpenCV REQUIRED)
# if(OpenCV_FOUND)
#     message(STATUS "OpenCV found")
#     message(${OpenCV_LIBS})
#     include_directories(${OpenCV_INCLUDE_DIRS})
#     target_link_libraries(yolov5 ${OpenCV_LIBS})
# endif()

find_package(onnxruntime)
if(onnxruntime_FOUND)
    message("onnxruntime found")
    message(${onnxruntime_LIBS})
    include_directories(${onnxruntime_INCLUDE_DIR})
    target_link_libraries(custom_op ${onnxruntime_LIBS})
endif()

# target_link_libraries(yolov5 ${OpenCV_LIBS} ${onnxruntime_LIBS})
# target_link_libraries(yolov5 /usr/local/lib/onnxruntime/libonnxruntime_providers_cuda.so)