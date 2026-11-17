set(TARGET autotuner)
add_executable(${TARGET} autotuner.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    add_test(NAME test-autotuner COMMAND $<TARGET_FILE:${TARGET}> --self-test)
endif()
