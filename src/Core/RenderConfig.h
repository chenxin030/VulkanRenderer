#pragma once

// RENDERING_LEVEL is set per-target via CMake (target_compile_definitions).
// If not defined, fall back to a default to keep legacy builds working.
#ifndef RENDERING_LEVEL
#define RENDERING_LEVEL 8
#endif
