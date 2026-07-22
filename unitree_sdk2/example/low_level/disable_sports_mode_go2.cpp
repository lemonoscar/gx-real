#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

namespace
{
constexpr int kMaxReleaseAttempts = 5;
constexpr auto kReleaseSettleTime = std::chrono::seconds(1);

int CheckMode(
    unitree::robot::b2::MotionSwitcherClient& client,
    std::string& robotForm,
    std::string& motionName)
{
    robotForm.clear();
    motionName.clear();
    const int32_t ret = client.CheckMode(robotForm, motionName);
    if (ret != 0)
    {
        std::cerr << "[gx-real] MotionSwitcherClient::CheckMode failed: " << ret << std::endl;
    }
    return ret;
}
}

int main(int argc, const char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " networkInterface [--require-active]" << std::endl;
        return EXIT_FAILURE;
    }

    const bool requireActive = argc == 3 && std::string(argv[2]) == "--require-active";
    if (argc == 3 && !requireActive)
    {
        std::cerr << "Unknown option: " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

        unitree::robot::b2::MotionSwitcherClient motionSwitcher;
        motionSwitcher.SetTimeout(10.0f);
        motionSwitcher.Init();

        std::string robotForm;
        std::string motionName;
        if (CheckMode(motionSwitcher, robotForm, motionName) != 0)
        {
            return EXIT_FAILURE;
        }
        if (requireActive)
        {
            if (motionName.empty())
            {
                std::cout << "[gx-real] motion-mode check: no active motion mode." << std::endl;
                if (requireActive)
                {
                    std::cerr << "[gx-real] calibration requires an active MCF motion mode; "
                              << "refusing unsupported standing calibration." << std::endl;
                    return EXIT_FAILURE;
                }
                return EXIT_SUCCESS;
            }
            std::cout << "[gx-real] active motion mode verified: " << motionName
                      << " (robot form " << robotForm << ")." << std::endl;
            return EXIT_SUCCESS;
        }
        if (motionName.empty())
        {
            std::cout << "[gx-real] MCF release verified: no active motion mode." << std::endl;
            return EXIT_SUCCESS;
        }

        std::cout << "WARNING: releasing MCF removes the built-in motion controller." << std::endl
                  << "Current motion mode: " << motionName
                  << " (robot form " << robotForm << ")." << std::endl
                  << "Make sure the robot is hung up or lying safely on the ground." << std::endl
                  << "Press Enter to release MCF, or Ctrl-C to abort." << std::endl;
        std::string confirmation;
        if (!std::getline(std::cin, confirmation))
        {
            std::cerr << "[gx-real] MCF release aborted: operator confirmation was not received."
                      << std::endl;
            return EXIT_FAILURE;
        }

        for (int attempt = 1; attempt <= kMaxReleaseAttempts; ++attempt)
        {
            const int32_t releaseRet = motionSwitcher.ReleaseMode();
            if (releaseRet != 0)
            {
                std::cerr << "[gx-real] MotionSwitcherClient::ReleaseMode failed on attempt "
                          << attempt << "/" << kMaxReleaseAttempts
                          << ": " << releaseRet << std::endl;
                std::this_thread::sleep_for(kReleaseSettleTime);
                continue;
            }

            std::this_thread::sleep_for(kReleaseSettleTime);
            if (CheckMode(motionSwitcher, robotForm, motionName) != 0)
            {
                return EXIT_FAILURE;
            }
            if (motionName.empty())
            {
                std::cout << "[gx-real] MCF release verified after attempt " << attempt << "."
                          << std::endl;
                return EXIT_SUCCESS;
            }

            std::cerr << "[gx-real] MCF is still active after attempt " << attempt << "/"
                      << kMaxReleaseAttempts << ": mode=" << motionName
                      << " form=" << robotForm << std::endl;
        }

        std::cerr << "[gx-real] Refusing low-level startup: MCF release could not be verified."
                  << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "[gx-real] MCF release failed with exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
}
