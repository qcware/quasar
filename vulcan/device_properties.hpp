#pragma once

#include <string>

namespace vulcan {

/// Number of devices that are visible to the CUDA runtime
int ndevice();

/**
 * A multiline string representing the properties of specified device ID.
 * Params:
 *  device (int) - device index in [0, ndevice)
 * Returns:
 *  (std::string) - multiline string with device name, memory size, an SM
 *   characteristics.
 **/
std::string device_property_string(int device);

} // namespace vulcan
