{ lib
, buildPythonPackage
, fetchPypi
, setuptools
, wheel
}:

buildPythonPackage rec {
  pname = "gym-retro";
  version = "0.8.0";
  
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-CP3V73yWSArRHBLUct4hrNMjWZlvaaUlkpm1QP66RWA=";
  };

  doCheck = false;

  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];

}
