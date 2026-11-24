# FILENAME: package.nix
{
  llvmPackages_23,
  cmake,
  pkg-config,
  gbenchmark,
  gtest,
  linuxHeaders,
  theta-cmake ? null,
  src,
}:
llvmPackages_23.stdenv.mkDerivation {
  inherit src;
  pname = "theta-hyper-shared-ptr";
  version = "0.2.0";

  outputs = [ "out" ];

  buildInputs = [
    gbenchmark
    gtest
    linuxHeaders
  ];

  propagatedBuildInputs = builtins.filter (x: x != null) [ theta-cmake ];

  nativeBuildInputs = [
    cmake
    pkg-config
    llvmPackages_23.clang-tools
  ];

  dontStrip = true;
  cmakeBuildType = "Release";
  separateDebugInfo = true;

  doCheck = false;
  env = {
    BUILD_BENCHMARK = "OFF";
  };
}
