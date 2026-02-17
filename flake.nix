{
  inputs = {
    pins.url = "git+https://github.com/positron-solutions/pins.git";
    nixpkgs.follows = "pins/nixpkgs";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      vulkanEnv = ''
          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d

          LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.pipewire}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.llvmPackages.libclang.lib}/lib:$LD_LIBRARY_PATH
      '';
      vulkanDeps = with pkgs; [
          vulkan-loader
          vulkan-headers
          vulkan-tools
          vulkan-validation-layers
          shader-slang
      ];

      pipewireEnv = ''
        LD_LIBRARY_PATH=${pkgs.pipewire}/lib:$LD_LIBRARY_PATH
      '';
      pipewireDeps = with pkgs; [
          pipewire
          llvmPackages.libclang.lib
      ];

      # `--release` binaries are built to have their assets paths baked into the
      # binary in order to make distribution reliable (by failing early).  This
      # variable is set in dev shells so that `cargo run --release` "just works."
      setAssetsDir = ''
        export MUTATE_ASSETS_DIR="$(realpath mutate-visualizer/assets)"
      '';

      x11Deps = with pkgs; [
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          libxkbcommon
      ];

      x11Shell = pkgs.mkShell {
        buildInputs = with pkgs; [
          pkg-config
        ] ++ pipewireDeps ++ x11Deps ++ vulkanDeps;

        # Make sure dynamic linker can find libX11 at runtime
        shellHook = ''
          ${setAssetsDir}
          ${vulkanEnv}
          ${pipewireEnv}
          LD_LIBRARY_PATH=${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.xorg.libXcursor}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.xorg.libXi}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH
          export LD_LIBRARY_PATH

          export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib
        '';
      };

      waylandDeps = with pkgs; [
        wayland
        wayland-protocols
        libxkbcommon
        libdecor
        libinput
      ];
      waylandShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          pkg-config
        ] ++ pipewireDeps ++ waylandDeps ++ vulkanDeps;

        # Make sure dynamic linker can find libX11 at runtime
        shellHook = ''
          ${setAssetsDir}
          ${vulkanEnv}
          ${pipewireEnv}
          LD_LIBRARY_PATH=${pkgs.wayland}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.wayland-cursor}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.libinput}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.libdecor}/lib:$LD_LIBRARY_PATH
          export LD_LIBRARY_PATH

          export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib
        '';
      };
    in {
      devShells = {
        x86_64-linux = {
          default = waylandShell;
          wayland = waylandShell;
          x11 = x11Shell;
        };
      };
    };
}
