cmake_minimum_required(VERSION 3.25)
project(tender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Third-party dependencies ---

include(FetchContent)

FetchContent_Declare(
    mpk_mix
    GIT_REPOSITORY https://github.com/mpksoft/mpk_mix.git
    GIT_TAG        ad48756c35d01ccab96f1ffa4858a358e9357fff
)
FetchContent_MakeAvailable(mpk_mix)

# --- tender library (to be populated) ---

# add_subdirectory(src)

# --- Tests ---

# add_subdirectory(tests)
