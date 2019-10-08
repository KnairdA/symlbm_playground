{ pkgs ? import <nixpkgs> { }, ... }:

pkgs.stdenvNoCC.mkDerivation rec {
  name = "pycl-env";
  env = pkgs.buildEnv { name = name; paths = buildInputs; };

  buildInputs = let
    custom-python = let
      packageOverrides = self: super: {
        pyopencl = super.pyopencl.overridePythonAttrs(old: rec {
          buildInputs = with pkgs; [
            opencl-headers ocl-icd python37Packages.pybind11
            libGLU_combined
          ];
        # Enable OpenGL integration and fix build
          preBuild = ''
            python configure.py --cl-enable-gl
            export HOME=/tmp/pyopencl
          '';
        });
      };
    in pkgs.python3.override { inherit packageOverrides; };

    pyevtk = pkgs.python3.pkgs.buildPythonPackage rec {
      pname = "PyEVTK";
      version = "1.2.1";

      src = pkgs.fetchFromGitHub {
        owner  = "paulo-herrera";
        repo   = "PyEVTK";
        rev    = "v1.2.1";
        sha256 = "1p2459dqvgakywvy5d31818hix4kic6ks9j4m582ypxyk5wj1ksz";
      };

      buildInputs = with pkgs.python37Packages; [
        numpy
      ];

      doCheck = false;
    };

    local-python = custom-python.withPackages (python-packages: with python-packages; [
      numpy
      sympy
      pyopencl
      pyopengl
      pyrr
      matplotlib
      Mako
      pyevtk

    # jupyter, optional
      jupyterlab
      ipywidgets
    ]);

  in [
    local-python
    pkgs.opencl-info
  # jupyter dependencies
    pkgs.nodejs

    pkgs.universal-ctags
  ];

  shellHook = ''
    export NIX_SHELL_NAME="${name}"
    export PYOPENCL_COMPILER_OUTPUT=1
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
}
