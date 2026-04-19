#include <unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp>

#include <iostream>
#include <string>

using namespace unitree::robot;
using namespace unitree::robot::go2;

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: obstacles_avoid_client_example [NetWorkInterface(eth0)] [on|off|status]" << std::endl;
        return 1;
    }

    std::string networkInterface = argv[1];
    std::string action = argc > 2 ? argv[2] : "off";

    if (action != "on" && action != "off" && action != "status")
    {
        std::cout << "Invalid action: " << action << std::endl;
        std::cout << "Usage: obstacles_avoid_client_example [NetWorkInterface(eth0)] [on|off|status]" << std::endl;
        return 1;
    }

    std::cout << "NetWorkInterface: " << networkInterface << std::endl;
    std::cout << "Action: " << action << std::endl;

    ChannelFactory::Instance()->Init(0, networkInterface);

    ObstaclesAvoidClient client;
    client.SetTimeout(10.0f);
    client.Init();

    bool enabled = false;
    int32_t ret = client.SwitchGet(enabled);
    std::cout << "SwitchGet ret: " << ret << ", enabled: " << enabled << std::endl;

    if (action != "status")
    {
        const bool target = action == "on";
        ret = client.SwitchSet(target);
        std::cout << "SwitchSet(" << target << ") ret: " << ret << std::endl;

        bool updated = false;
        ret = client.SwitchGet(updated);
        std::cout << "SwitchGet ret: " << ret << ", enabled: " << updated << std::endl;
    }

    ChannelFactory::Instance()->Release();
    return 0;
}
