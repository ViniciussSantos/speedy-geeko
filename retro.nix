{ buildPythonPackage
, setuptools
, fetchFromGitHub, cmake, gcc, setuptools-scm, zlib, pkg-config, dbus, wheel, gym, gym-notices, pyglet, numpy, cloudpickle}:

buildPythonPackage rec {
  pname = "retro";
  version = "0.8.0";
  format = "setuptools";
  
  src = fetchFromGitHub {
    owner = "openai";
    repo = pname; 
    rev = "refs/tags/v${version}";
    hash = "sha256-JiMNkHb5NrQREuwmwoFrRCd1Zs/gtCgwx2ElgQx52XA=";
  };

  doCheck = false;

  build-system = [
   setuptools
   setuptools-scm
  ];

  buildInputs = [
    dbus
  ];

  nativeBuildInputs = [
    gcc
    cmake
    pkg-config
    wheel
  ];

  dependencies = [
    zlib
    gym
    pyglet
    gym-notices
    numpy
    cloudpickle
  ];

  pythonImportsCheck = [ "gym" ];
}
