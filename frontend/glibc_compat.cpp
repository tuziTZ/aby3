#include <cstdlib>

// Some vendored static libraries were built against a newer glibc that emits
// C23 symbol names. Keep local Ubuntu 22.04 builds linkable without rebuilding
// the whole third-party stack.
extern "C" unsigned long long __attribute__((weak))
__isoc23_strtoull(const char* nptr, char** endptr, int base)
{
    return std::strtoull(nptr, endptr, base);
}
