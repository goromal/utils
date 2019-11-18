# Utils

Header-only library with common C++ utilities.

To include in your project, add the following lines to your *CMakeLists.txt* file:

```cmake
if (NOT TARGET utils)
  include_directories(path/to/utils/include)
  add_subdirectory(path/to/lib/utils)
endif()
```
