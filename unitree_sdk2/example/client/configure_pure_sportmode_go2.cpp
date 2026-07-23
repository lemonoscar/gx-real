#include <unitree/common/json/jsonize.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp>
#include <unitree/robot/go2/sport/sport_api.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/utrack/utrack_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace {

constexpr int32_t kApiHandStand = 2044;
constexpr int32_t kApiFreeBound = 2046;
constexpr int32_t kApiFreeJump = 2047;
constexpr int32_t kApiFreeAvoid = 2048;
constexpr int32_t kApiClassicWalk = 2049;
constexpr int32_t kApiWalkUpright = 2050;
constexpr int32_t kApiCrossStep = 2051;
constexpr int32_t kApiAutoRecoverySet = 2054;
constexpr int32_t kApiAutoRecoveryGet = 2055;

class BooleanData final : public unitree::common::Jsonize {
 public:
  void fromJson(unitree::common::JsonMap& json) override {
    unitree::common::FromJson(json["data"], value);
  }

  void toJson(unitree::common::JsonMap& json) const override {
    unitree::common::ToJson(value, json["data"]);
  }

  bool value = true;
};

class ExtendedSportFeatureClient final : public unitree::robot::Client {
 public:
  ExtendedSportFeatureClient()
      : unitree::robot::Client(
            unitree::robot::go2::ROBOT_SPORT_SERVICE_NAME) {}

  void Init() override {
    SetApiVersion(unitree::robot::go2::ROBOT_SPORT_API_VERSION);
    RegistApi(kApiHandStand);
    RegistApi(kApiFreeBound);
    RegistApi(kApiFreeJump);
    RegistApi(kApiFreeAvoid);
    RegistApi(kApiClassicWalk);
    RegistApi(kApiWalkUpright);
    RegistApi(kApiCrossStep);
    RegistApi(kApiAutoRecoverySet);
    RegistApi(kApiAutoRecoveryGet);
  }

  int32_t Disable(int32_t api_id) {
    return Call(api_id, R"({"data":false})");
  }

  int32_t GetAutoRecovery(bool& enabled) {
    std::string data;
    const int32_t ret = Call(kApiAutoRecoveryGet, "", data);
    if (ret != 0) {
      return ret;
    }
    try {
      BooleanData state;
      unitree::common::FromJsonString(data, state);
      enabled = state.value;
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "[pure-sportmode] invalid AutoRecoveryGet response: "
                << error.what() << std::endl;
      return -1;
    } catch (...) {
      std::cerr << "[pure-sportmode] invalid AutoRecoveryGet response"
                << std::endl;
      return -1;
    }
  }
};

void RecordResult(const std::string& action, int32_t ret, int& failures) {
  if (ret == 0) {
    std::cout << "[pure-sportmode] OK: " << action << std::endl;
    return;
  }
  ++failures;
  std::cerr << "[pure-sportmode] FAILED: " << action
            << " (SDK code " << ret << ")" << std::endl;
}

void RecordIdempotentResult(const std::string& action, int32_t ret,
                            int& failures) {
  if (ret != -1) {
    RecordResult(action, ret, failures);
    return;
  }
  std::cerr << "[pure-sportmode] WARNING: " << action
            << " returned SDK code -1; accepted as an already inactive "
               "idempotent no-op on this firmware"
            << std::endl;
}

int32_t WaitForUtrackInactive(unitree::robot::go2::UtrackClient& utrack,
                              bool& tracking) {
  for (int attempt = 0; attempt < 10; ++attempt) {
    const int32_t ret = utrack.IsTracking(tracking);
    if (ret != 0 || !tracking) {
      return ret;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return -1;
}

}  // namespace

int main(int argc, const char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: configure_pure_sportmode_go2 NETWORK_INTERFACE"
              << std::endl;
    return 2;
  }

  const std::string network_interface = argv[1];
  unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);

  unitree::robot::go2::SportClient sport;
  sport.SetTimeout(5.0f);
  sport.Init();

  unitree::robot::go2::ObstaclesAvoidClient obstacles;
  obstacles.SetTimeout(5.0f);
  obstacles.Init();

  unitree::robot::go2::UtrackClient utrack;
  utrack.SetTimeout(5.0f);
  utrack.Init();

  unitree::robot::go2::VuiClient vui;
  vui.SetTimeout(5.0f);
  vui.Init();

  ExtendedSportFeatureClient extended;
  extended.SetTimeout(5.0f);
  extended.Init();

  int failures = 0;
  RecordIdempotentResult("StopMove before configuration", sport.StopMove(),
                         failures);
  RecordResult("light brightness=0 before configuration",
               vui.SetBrightness(0), failures);

  RecordResult("obstacle avoidance=false", obstacles.SwitchSet(false), failures);
  bool obstacle_avoidance_enabled = true;
  const int32_t obstacle_get_ret =
      obstacles.SwitchGet(obstacle_avoidance_enabled);
  RecordResult("read obstacle avoidance state", obstacle_get_ret, failures);
  if (obstacle_get_ret == 0 && obstacle_avoidance_enabled) {
    ++failures;
    std::cerr << "[pure-sportmode] FAILED: obstacle avoidance remained enabled"
              << std::endl;
  }

  RecordResult("UWB tracking switch=false", utrack.SwitchSet(false), failures);
  bool utrack_enabled = true;
  const int32_t utrack_get_ret = utrack.SwitchGet(utrack_enabled);
  RecordResult("read UWB tracking switch", utrack_get_ret, failures);
  if (utrack_get_ret == 0 && utrack_enabled) {
    ++failures;
    std::cerr << "[pure-sportmode] FAILED: UWB tracking switch remained enabled"
              << std::endl;
  }
  bool utrack_tracking = true;
  const int32_t tracking_ret = WaitForUtrackInactive(utrack, utrack_tracking);
  RecordResult("confirm UWB tracking inactive", tracking_ret, failures);
  if (tracking_ret == 0 && utrack_tracking) {
    ++failures;
    std::cerr << "[pure-sportmode] FAILED: UWB tracking remained active"
              << std::endl;
  }

  RecordResult("firmware joystick arbitration=false",
               sport.SwitchJoystick(false), failures);
  RecordIdempotentResult("pose mode=false", sport.Pose(false), failures);

  RecordResult("hand stand=false", extended.Disable(kApiHandStand), failures);
  RecordResult("free bound=false", extended.Disable(kApiFreeBound), failures);
  RecordResult("free jump=false", extended.Disable(kApiFreeJump), failures);
  RecordResult("free avoid=false", extended.Disable(kApiFreeAvoid), failures);
  RecordResult("classic walk=false", extended.Disable(kApiClassicWalk), failures);
  RecordResult("walk upright=false", extended.Disable(kApiWalkUpright), failures);
  RecordResult("cross step=false", extended.Disable(kApiCrossStep), failures);
  RecordResult("auto recovery=false", extended.Disable(kApiAutoRecoverySet),
               failures);

  bool auto_recovery_enabled = true;
  const int32_t recovery_get_ret =
      extended.GetAutoRecovery(auto_recovery_enabled);
  RecordResult("read auto recovery state", recovery_get_ret, failures);
  if (recovery_get_ret == 0 && auto_recovery_enabled) {
    ++failures;
    std::cerr << "[pure-sportmode] FAILED: auto recovery remained enabled"
              << std::endl;
  }

  RecordIdempotentResult("StopMove after configuration", sport.StopMove(),
                         failures);
  RecordResult("light brightness=0 after configuration",
               vui.SetBrightness(0), failures);
  int brightness = -1;
  const int32_t brightness_get_ret = vui.GetBrightness(brightness);
  RecordResult("read light brightness", brightness_get_ret, failures);
  if (brightness_get_ret == 0 && brightness != 0) {
    ++failures;
    std::cerr << "[pure-sportmode] FAILED: light brightness remained "
              << brightness << " instead of 0" << std::endl;
  }
  unitree::robot::ChannelFactory::Instance()->Release();

  if (failures != 0) {
    std::cerr << "[pure-sportmode] refusing startup: " << failures
              << " SDK configuration check(s) failed" << std::endl;
    return 1;
  }

  std::cout << "[pure-sportmode] required configuration checks passed; "
               "readable states and light brightness confirmed at 0"
            << std::endl;
  return 0;
}
