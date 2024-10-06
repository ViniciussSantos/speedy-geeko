let
  pkgs = import <nixpkgs> {};

  python = pkgs.python39.override {
    self = python;
    packageOverrides = pyfinal: pyprev: {
      retro = pyfinal.callPackage ./retro.nix { };
    };
  };

in pkgs.mkShell {
  packages = [
    (python.withPackages (python-pkgs: [
      python-pkgs.retro
    ]))
  ];
}
