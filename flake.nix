{
  inputs = {
    pins.url = "git+https://github.com/positron-solutions/pins.git";
    nixpkgs.follows = "pins/nixpkgs";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      x11Shell = pkgs.mkShell {
        buildInputs = with pkgs; [
          pkg-config
          vulkan-loader
          vulkan-headers
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          libxkbcommon

          shader-slang

          pipewire
          llvmPackages.libclang.lib

          vulkan-loader
          vulkan-tools
          vulkan-validation-layers
        ];

        # Make sure dynamic linker can find libX11 at runtime
        shellHook = ''
          LD_LIBRARY_PATH=${pkgs.xorg.libX11}/lib
          LD_LIBRARY_PATH=${pkgs.xorg.libXcursor}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.xorg.libXi}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.pipewire}/lib:$LD_LIBRARY_PATH
          LD_LIBRARY_PATH=${pkgs.llvmPackages.libclang.lib}/lib:$LD_LIBRARY_PATH
          export LD_LIBRARY_PATH

          export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib

          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
        '';
      };
    in {
      devShells = {
        x86_64-linux = {
          default = x11Shell;
        };
      };
    };
}
