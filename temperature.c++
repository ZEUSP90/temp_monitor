/*
 * temp_monitor.cpp
 * Dead-accurate CPU + NVIDIA GPU temperature monitor for Linux
 *
 * CPU  : reads directly from /sys/class/hwmon/ (coretemp kernel driver)
 * GPU  : uses NVML (libnvidia-ml) — same source as nvidia-smi
 *
 * Compile:
 *   g++ -O2 -o temp_monitor temp_monitor.cpp -ldl
 *
 * Run:
 *   sudo ./temp_monitor          (sudo needed for full CPU core access)
 *
 * Requirements:
 *   - NVIDIA driver installed (libnvidia-ml.so ships with it)
 *   - kernel modules loaded: coretemp (Intel) OR k10temp (AMD CPU)
 *     check with: lsmod | grep -E 'coretemp|k10temp'
 *     load with:  sudo modprobe coretemp   (or k10temp)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <dlfcn.h>       // dlopen / dlsym — load NVML at runtime
#include <thread>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <cstring>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  NVML types / constants (subset we need)
//  Defined here so we don't need the CUDA SDK headers
// ─────────────────────────────────────────────
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0

typedef nvmlReturn_t (*pfn_nvmlInit)(void);
typedef nvmlReturn_t (*pfn_nvmlShutdown)(void);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount)(unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);

struct NvmlLib {
    void*                           handle = nullptr;
    pfn_nvmlInit                    Init = nullptr;
    pfn_nvmlShutdown                Shutdown = nullptr;
    pfn_nvmlDeviceGetCount          DeviceGetCount = nullptr;
    pfn_nvmlDeviceGetHandleByIndex  DeviceGetHandleByIndex = nullptr;
    pfn_nvmlDeviceGetTemperature    DeviceGetTemperature = nullptr;
    pfn_nvmlDeviceGetName           DeviceGetName = nullptr;
    bool ok = false;
};

// ─────────────────────────────────────────────
//  Structs
// ─────────────────────────────────────────────
struct CpuSensor {
    std::string label;   // e.g. "Package id 0", "Core 0"
    std::string path;    // /sys/class/hwmon/hwmonX/tempY_input
};

struct GpuInfo {
    std::string name;
    nvmlDevice_t handle;
};

// ─────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────
static volatile bool g_running = true;
static void on_signal(int) { g_running = false; }

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::string s;
    std::getline(f, s);
    return s;
}

// Reads millidegree value and converts to °C
static double read_temp_c(const std::string& path) {
    std::string raw = read_file(path);
    if (raw.empty()) return -1.0;
    try {
        return std::stod(raw) / 1000.0;
    } catch (...) {
        return -1.0;
    }
}

// ─────────────────────────────────────────────
//  Discover CPU sensors via hwmon
// ─────────────────────────────────────────────
static std::vector<CpuSensor> find_cpu_sensors() {
    std::vector<CpuSensor> sensors;
    const std::string base = "/sys/class/hwmon";

    if (!fs::exists(base)) {
        std::cerr << "[WARN] /sys/class/hwmon not found. Load coretemp/k10temp module.\n";
        return sensors;
    }

    for (auto& hwmon : fs::directory_iterator(base)) {
        std::string name_path = hwmon.path().string() + "/name";
        std::string chip_name = read_file(name_path);

        // coretemp = Intel CPU, k10temp = AMD CPU
        if (chip_name != "coretemp" && chip_name != "k10temp" &&
            chip_name != "zenpower" && chip_name != "nct6779" &&
            chip_name != "it8686") {
            // Still try if it has temp inputs — catches edge cases
        }

        // Scan tempN_input files
        for (auto& entry : fs::directory_iterator(hwmon.path())) {
            std::string fname = entry.path().filename().string();
            // Match tempN_input
            if (fname.size() > 10 &&
                fname.substr(0, 4) == "temp" &&
                fname.substr(fname.size() - 6) == "_input") {

                std::string prefix = entry.path().string().substr(
                    0, entry.path().string().size() - 6); // strip "_input"

                std::string label_path = prefix + "_label";
                std::string label = read_file(label_path);
                if (label.empty()) label = fname; // fallback to filename

                // Filter: only CPU package + cores
                bool is_cpu_sensor =
                    (chip_name == "coretemp" || chip_name == "k10temp" ||
                     chip_name == "zenpower") ||
                    label.find("Package") != std::string::npos ||
                    label.find("Core")    != std::string::npos ||
                    label.find("Tdie")    != std::string::npos ||
                    label.find("Tccd")    != std::string::npos;

                if (is_cpu_sensor) {
                    sensors.push_back({ "[" + chip_name + "] " + label,
                                        entry.path().string() });
                }
            }
        }
    }
    return sensors;
}

// ─────────────────────────────────────────────
//  Load NVML at runtime (no CUDA SDK needed)
// ─────────────────────────────────────────────
static NvmlLib load_nvml() {
    NvmlLib lib;
    // Try both versioned and unversioned sonames
    const char* names[] = {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        nullptr
    };
    for (int i = 0; names[i]; i++) {
        lib.handle = dlopen(names[i], RTLD_LAZY);
        if (lib.handle) break;
    }
    if (!lib.handle) {
        std::cerr << "[WARN] Could not load libnvidia-ml.so. "
                     "Is the NVIDIA driver installed?\n";
        return lib;
    }

#define LOAD(fn) lib.fn = (pfn_nvml##fn)dlsym(lib.handle, "nvml" #fn); \
    if (!lib.fn) { std::cerr << "[WARN] nvml" #fn " not found\n"; return lib; }

    LOAD(Init)
    LOAD(Shutdown)
    LOAD(DeviceGetCount)
    LOAD(DeviceGetHandleByIndex)
    LOAD(DeviceGetTemperature)
    LOAD(DeviceGetName)
#undef LOAD

    if (lib.Init() != NVML_SUCCESS) {
        std::cerr << "[WARN] nvmlInit() failed.\n";
        return lib;
    }
    lib.ok = true;
    return lib;
}

// ─────────────────────────────────────────────
//  Discover NVIDIA GPUs
// ─────────────────────────────────────────────
static std::vector<GpuInfo> find_gpus(NvmlLib& nvml) {
    std::vector<GpuInfo> gpus;
    if (!nvml.ok) return gpus;

    unsigned int count = 0;
    if (nvml.DeviceGetCount(&count) != NVML_SUCCESS) return gpus;

    for (unsigned int i = 0; i < count; i++) {
        GpuInfo g;
        if (nvml.DeviceGetHandleByIndex(i, &g.handle) != NVML_SUCCESS) continue;
        char name[96] = {};
        nvml.DeviceGetName(g.handle, name, sizeof(name));
        g.name = name;
        gpus.push_back(g);
    }
    return gpus;
}

// ─────────────────────────────────────────────
//  Display helpers
// ─────────────────────────────────────────────
static std::string temp_bar(double t, double max = 100.0) {
    int filled = (int)((t / max) * 20.0);
    if (filled < 0) filled = 0;
    if (filled > 20) filled = 20;
    std::string bar = "[";
    for (int i = 0; i < 20; i++) bar += (i < filled ? "█" : "░");
    bar += "]";
    return bar;
}

static std::string color_temp(double t) {
    // ANSI colors: green < 60, yellow < 80, red >= 80
    if (t < 60.0)  return "\033[32m"; // green
    if (t < 80.0)  return "\033[33m"; // yellow
    return "\033[31m";                 // red
}
const char* RESET = "\033[0m";
const char* BOLD  = "\033[1m";
const char* CYAN  = "\033[36m";
const char* DIM   = "\033[2m";

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::cout << BOLD << CYAN
              << "╔══════════════════════════════════════════╗\n"
              << "║     Linux CPU + GPU Temperature Monitor  ║\n"
              << "║     Source: hwmon kernel driver + NVML   ║\n"
              << "╚══════════════════════════════════════════╝\n"
              << RESET << "\n";

    // --- Discover sensors ---
    auto cpu_sensors = find_cpu_sensors();
    if (cpu_sensors.empty()) {
        std::cout << "[INFO] No CPU sensors found. Try:\n"
                  << "       sudo modprobe coretemp   # Intel\n"
                  << "       sudo modprobe k10temp    # AMD\n\n";
    }

    NvmlLib nvml = load_nvml();
    auto gpus = find_gpus(nvml);
    if (gpus.empty() && nvml.ok) {
        std::cout << "[INFO] NVML loaded but no NVIDIA GPUs found.\n\n";
    }

    if (cpu_sensors.empty() && gpus.empty()) {
        std::cerr << "[ERROR] No sensors found at all. Exiting.\n";
        return 1;
    }

    std::cout << DIM << "Refreshing every 1 second. Press Ctrl+C to quit.\n\n" << RESET;

    int refresh_ms = 1000;

    while (g_running) {
        // Move cursor up to overwrite previous output after first print
        static int lines_printed = 0;
        if (lines_printed > 0) {
            // Move up `lines_printed` lines
            std::cout << "\033[" << lines_printed << "A";
        }
        lines_printed = 0;

        auto print_line = [&](const std::string& s) {
            // Pad to 60 chars to clear any leftover chars, then newline
            std::cout << std::left << std::setw(60) << s << "\n";
            lines_printed++;
        };

        // Timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        char tbuf[32];
        std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", std::localtime(&t));
        print_line(std::string(BOLD) + "  Updated: " + tbuf + RESET);
        print_line("");

        // ── CPU ──
        if (!cpu_sensors.empty()) {
            print_line(std::string(BOLD) + CYAN + "  ── CPU ──" + RESET);
            for (auto& s : cpu_sensors) {
                double temp = read_temp_c(s.path);
                if (temp < 0) continue;
                std::ostringstream line;
                line << "  " << std::left << std::setw(30) << s.label
                     << color_temp(temp)
                     << std::fixed << std::setprecision(1)
                     << std::setw(6) << temp << " °C  "
                     << temp_bar(temp)
                     << RESET;
                print_line(line.str());
            }
            print_line("");
        }

        // ── GPU ──
        if (!gpus.empty()) {
            print_line(std::string(BOLD) + CYAN + "  ── GPU ──" + RESET);
            for (size_t i = 0; i < gpus.size(); i++) {
                unsigned int gpu_temp = 0;
                nvmlReturn_t r = nvml.DeviceGetTemperature(
                    gpus[i].handle, NVML_TEMPERATURE_GPU, &gpu_temp);
                double temp = (r == NVML_SUCCESS) ? (double)gpu_temp : -1.0;

                std::ostringstream line;
                line << "  GPU " << i << ": " << std::left << std::setw(24) << gpus[i].name;
                if (temp >= 0) {
                    line << color_temp(temp)
                         << std::fixed << std::setprecision(1)
                         << std::setw(6) << temp << " °C  "
                         << temp_bar(temp)
                         << RESET;
                } else {
                    line << "  N/A";
                }
                print_line(line.str());
            }
            print_line("");
        }

        std::cout << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
    }

    std::cout << "\n\033[0m[INFO] Exiting...\n";
    if (nvml.ok) nvml.Shutdown();
    if (nvml.handle) dlclose(nvml.handle);
    return 0;
}