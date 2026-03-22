# Project Context
 
- Vulkan 1.4: dynamic rendering, bindless descriptor table, buffer device address.
- `ash` bindings for Rust. Shader language is Slang. Window integration is `winit`.

## Macros & Codegen

- Slang introspection JSON feeds proc macro codegen.
- Pre-baked witness data drives runtime technique resolution.

## API Style

- Reducing Vulkan to an ergonomic subset.
- Builder semantics via `bon` to preserve some flexibility without taxing callers.

## Project Status

- Still developing concrete types for macros to build on top of, even getting struct fields into the right combinations.
- Avoid RAII or complex lifetime coupling.  Just use manual destruction until other details emerge.

