project(helloworld)
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_STANDARD 17)

include_directories(include thirdparty/include thirdparty/include/ddscxx thirdparty/include/iceoryx/v2.0.2)
link_directories(lib/${CMAKE_HOST_SYSTEM_PROCESSOR} thirdparty/lib/${CMAKE_HOST_SYSTEM_PROCESSOR})

link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)

add_executable(test_publisher example/helloworld/publisher.cpp example/helloworld/HelloWorldData.cpp)
add_executable(test_subscriber example/helloworld/subscriber.cpp example/helloworld/HelloWorldData.cpp)
add_executable(high_follow_sin example/high_level/follow_sin.cpp)
add_executable(sportmode_test example/high_level/sportmode_test.cpp)
add_executable(low_level example/low_level/low_level.cpp)
add_executable(stand_example_go2 example/low_level/stand_example_go2.cpp)
add_executable(disable_sports_mode_go2 example/low_level/disable_sports_mode_go2.cpp)
add_executable(stand_example_b2  example/low_level/stand_example_b2.cpp)
add_executable(wireless example/wireless/wireless.cpp)
add_executable(advanced_gamepad example/advanced_gamepad/main.cpp)
add_executable(state_machine_example example/state_machine/main.cpp)
add_executable(test_jsonize example/jsonize/test_jsonize.cpp)
add_executable(robot_state_client_example example/client/robot_state_client_example.cpp)
add_executable(sport_client_example example/client/sport_client_example.cpp)
add_executable(obstacles_avoid_client_example example/client/obstacles_avoid_client_example.cpp)
add_executable(configure_pure_sportmode_go2 example/client/configure_pure_sportmode_go2.cpp)
set_target_properties(configure_pure_sportmode_go2 PROPERTIES BUILD_RPATH "\$ORIGIN")
add_custom_command(TARGET configure_pure_sportmode_go2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libddsc.so
          $<TARGET_FILE_DIR:configure_pure_sportmode_go2>/libddsc.so.0
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libddscxx.so
          $<TARGET_FILE_DIR:configure_pure_sportmode_go2>/libddscxx.so.0)
add_executable(video_client_example example/client/video_client_example.cpp)
add_executable(vui_client_example example/client/vui_client_example.cpp)
add_executable(h1_arm_sdk_dds_example example/humanoid/high_level/h1_arm_sdk_dds_example.cpp)
add_executable(h1_loco_example_client example/humanoid/high_level/h1_loco_example_client.cpp)
add_executable(h1_low_level_example example/humanoid/low_level/humanoid.cpp)
