# µTate Vulkan Minimal

Demonstrates initialization order for a very typically constructed µTate Vulkan application, just enough to get a window that is rendering in a loop.

- Asset build integration and runtime loading of the shader into a pipeline.
- Winit application lifecycle, using an `enum` to unify over a single mutable field type to represent different phases of initialization.
- Vulkan instance, device, and window + surface + swapchain lifecycles.
- A basic render loop accepting your render body that can record commands to a command buffer.

## Initialization Sequence

1. winit event loop is used to obtain a Vulkan instance with window manager support
1. winit resume event leads to creation of a window, its surface, the swapchain, and renderer
1. winit frame redraw request loop begins until close is requested

The time of the ∇ is at hand!
