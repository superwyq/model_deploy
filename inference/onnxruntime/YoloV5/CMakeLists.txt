cmake_minimum_required(VERSION 3.5)
PROJECT(yolov5)

set(CMAKE_BUILD_TYPE "Debug")


# set(OnnxRuntime_INCLUDE_DIR /usr/local/include/onnxruntime)

# include_directories(${OpenCV_INCLUDE_DIR} ${OnnxRuntime_INCLUDE_DIR})

add_executable(yolov5 main.cpp)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(${OpenCV_LIBS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolov5 ${OpenCV_LIBS})
endif()

find_package(onnxruntime)
if(onnxruntime_FOUND)
    message("onnxruntime found")
    message(${onnxruntime_LIBS})
    include_directories(${onnxruntime_INCLUDE_DIR})
    target_link_libraries(yolov5 ${onnxruntime_LIBS})
endif()

# target_link_libraries(yolov5 ${OpenCV_LIBS} ${onnxruntime_LIBS})
# target_link_libraries(yolov5 /usr/local/lib/onnxruntime/libonnxruntime_providers_cuda.so)