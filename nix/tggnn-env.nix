{ pkgs }:
let
  python = pkgs.python39;
  py-ver = builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;

  pytorch-1100 = python.pkgs.buildPythonPackage rec {
    version = "1.10.0";

    pname = "pytorch";

    format = "wheel";

    src = pkgs.fetchurl {
      name = "torch-${version}-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
      url =
        "https://download.pytorch.org/whl/cpu/torch-${version}%2Bcpu-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
      hash = "sha256-AhrcHXd3YiCsId+p/H3gi3H/OskHL77ELw8r/mIWheE=";
    };

    nativeBuildInputs = with pkgs; [ addOpenGLRunpath patchelf ];

    propagatedBuildInputs = with python.pkgs; [
      future
      numpy
      pyyaml
      requests
      typing-extensions
    ];

    postInstall = ''
      # ONNX conversion
      rm -rf $out/bin
    '';

    postFixup = let rpath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
    in ''
      find $out/${python.sitePackages}/torch/lib -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
        echo "setting rpath for $lib..."
        patchelf --set-rpath "${rpath}:$out/${python.sitePackages}/torch/lib" "$lib"
        addOpenGLRunpath "$lib"
      done
    '';

    pythonImportsCheck = [ "torch" ];

    meta = with pkgs.lib; {
      description =
        "Open source, prototype-to-production deep learning platform";
      homepage = "https://pytorch.org/";
      changelog = "https://github.com/pytorch/pytorch/releases/tag/v${version}";
      license = licenses.unfree; # Includes CUDA and Intel MKL.
      platforms = platforms.linux;
      maintainers = with maintainers; [ danieldk ];
    };
  };
  sparse = with python.pkgs;
    buildPythonPackage rec {
      pname = "torch_sparse";
      version = "0.6.12";

      src = pkgs.fetchurl {
        url =
          "https://data.pyg.org/whl/torch-${pytorch-1100.version}+cpu/${pname}-${version}-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
        hash = "sha256-s52CCmQP+V1P+4iDomm7ZRK0crAWiNwp7wGjrOLPn64=";
      };

      format = "wheel";

      propagatedBuildInputs = [ pytorch-1100 scipy ];

      doCheck = false;

      postInstall = ''
        rm -rf $out/${python.sitePackages}/test
      '';
    };
  scatter = with python.pkgs;
    buildPythonPackage rec {
      pname = "torch_scatter";
      version = "2.0.9";

      src = pkgs.fetchurl {
        name = "${pname}-${version}-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
        url =
          "https://data.pyg.org/whl/torch-${pytorch-1100.version}+cpu/${pname}-${version}-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
        hash = "sha256-WUl1SGcKrv6dN7q2m2qUurS9gFAByc2FKKXj1rP8fi4=";
      };

      format = "wheel";

      propagatedBuildInputs = [ pytorch-1100 ];

      doCheck = false;

      postInstall = ''
        rm -rf $out/${python.sitePackages}/test
      '';
    };
  tggnn = with python.pkgs;
    buildPythonPackage rec {
      pname = "tggnn";
      version = "0.3.0";

      src = ./..;

      propagatedBuildInputs = [ pytorch-1100 jsonpickle scatter ];

      doCheck = false;
    };
  iopath = with python.pkgs;
    buildPythonPackage rec {
      pname = "iopath";
      version = "0.1.9";

      format = "wheel";

      src = fetchPypi {
        inherit pname version format;
        dist = "py3";
        python = "py3";
        sha256 = "sha256-kFisJPAyjezfjb4gmzMHS4cCo8TZui94AdRsthqIC24=";
      };

      propagatedBuildInputs = [ tqdm portalocker ];

      doCheck = false;
    };
  fvcore = with python.pkgs;
    buildPythonPackage rec {
      pname = "fvcore";
      version = "0.1.5.post20220119";

      src = fetchPypi {
        inherit pname version;
        sha256 = "sha256-N+ca7Uf/Q56JmjBsJ+5r7prabVIsHaxOcMIUu7LG49c=";
      };

      propagatedBuildInputs =
        [ yacs tqdm pillow termcolor tabulate numpy iopath portalocker ];

      doCheck = false;
    };
  pytorch-3d = with python.pkgs;
    buildPythonPackage rec {
      pname = "pytorch3d";
      version = "0.6.1";

      src = pkgs.fetchurl {
        url =
          "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v${version}.tar.gz";
        sha256 = "sha256-pmVCusREBtj9nw15HYR7mvgW1n84IekekUZGwX6vvz8=";
      };

      nativeBuildInputs = with pkgs; [ which ];
      propagatedBuildInputs = [ fvcore iopath pytorch-1100 ];

      doCheck = false;
    };
  # pytorch-3d = with python.pkgs;
  # buildPythonPackage rec {
  #   pname = "pytorch3d";
  #   version = "0.6.1";

  #   format = "wheel";

  #   src = pkgs.fetchurl {
  #     url =
  #       "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu102_pyt1100/${pname}-${version}-cp${py-ver}-cp${py-ver}-linux_x86_64.whl";
  #     sha256 = "sha256-C3P4wnwDzL43P/EKnNloSEuGgXjttg39wFSmGcJzzhg=";
  #   };

  #   propagatedBuildInputs = [ fvcore iopath ];

  #   doCheck = false;
  # };
  python-env = python.withPackages (ps:
    with ps; [
      pytorch-1100
      pytorch-3d
      scatter
      sparse
      tggnn
      black
      matplotlib
      numpy-stl
      scipy
      tensorflow
      more-itertools
      jsonpickle
      pylint
    ]);
in pkgs.mkShell {
  name = "tggnn-env";
  buildInputs = [ python-env pkgs.pyright ];
}
