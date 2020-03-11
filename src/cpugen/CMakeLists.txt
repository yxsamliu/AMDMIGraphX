
add_executable(cpugen 
    main.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(cpugen migraphx_onnx migraphx_tf ${CMAKE_THREAD_LIBS_INIT})
