PROJECT(mnist)

set(CMAKE_BUILD_TYPE "Debug")

set(OpenCV_INCLUDE_DIR /usr/local/include/opencv)
set(OnnxRuntime_INCLUDE_DIR /usr/local/include/onnxruntime)
set(OpenCV_LIB_DIR /usr/local/lib)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)

include_directories(${OpenCV_INCLUDE_DIR} ${OnnxRuntime_INCLUDE_DIR})

add_executable(mnist infer.cpp)
target_link_libraries(mnist ${OpenCV_LIBS} /usr/local/lib/onnxruntime/libonnxruntime.so)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(mnist ${OpenCV_LIBS})
endif()
