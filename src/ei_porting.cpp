#include <Arduino.h>
#include <stdlib.h>
#include <stdint.h>

// Edge Impulse expects these symbols
extern "C" {

// Printf used by EI
void ei_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    char buf[256];
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    Serial.print(buf);
}

// Some EI code uses this
void ei_printf_float(float f) {
    Serial.print(f);
}

// Timing (microseconds)
uint64_t ei_read_timer_us() {
    return (uint64_t)esp_timer_get_time();
}

// If you don't implement cancellation, just return false (0)
int ei_run_impulse_check_canceled() {
    return 0;
}

// Memory allocation wrappers
void* ei_malloc(size_t size) {
    return malloc(size);
}

void* ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

void ei_free(void* ptr) {
    free(ptr);
}

} // extern "C"