#this file is only an example to configure CMAKE_TOOLCHAIN_FILE for cross compile CNStream

SET(CMAKE_SYSTEM_NAME Linux)

SET(CROSS_PREFIX aarch64-linux-gnu-)  # for live555
SET(CMAKE_C_COMPILER  aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
