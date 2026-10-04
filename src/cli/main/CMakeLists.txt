###############################
###           cli           ###
###############################

find_package(spdlog REQUIRED)
find_package(cli11 REQUIRED)

add_executable(cli
    main.cpp
    validators.cpp
)

add_executable(HBM2_runner
    HBM2_runner.cpp
)

target_compile_features(HBM2_runner PUBLIC cxx_std_17)
set_target_properties(HBM2_runner PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(HBM2_runner PROPERTIES CXX_STANDARD_REQUIRED ON)

target_link_libraries(HBM2_runner
PRIVATE
    DRAMPower::cli_lib
    DRAMPower::DRAMPower
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

add_executable(HBM3_runner
    HBM3_runner.cpp
)

target_compile_features(HBM3_runner PUBLIC cxx_std_17)
set_target_properties(HBM3_runner PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(HBM3_runner PROPERTIES CXX_STANDARD_REQUIRED ON)

target_link_libraries(HBM3_runner
PRIVATE
    DRAMPower::cli_lib
    DRAMPower::DRAMPower
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

target_compile_features(cli PUBLIC cxx_std_17)
set_target_properties(cli PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(cli PROPERTIES CXX_STANDARD_REQUIRED ON)

target_link_libraries(cli 
PRIVATE
    DRAMPower::cli_lib
    DRAMPower::DRAMPower
    spdlog::spdlog
    CLI11::CLI11
)
target_compile_definitions(cli PRIVATE DRAMPOWER_VERSION_STRING="${DRAMPOWER_VERSION_STRING}")