# FILENAME: treefmt.nix
{ pkgs, ... }:
{
  projectRootFile = "flake.nix";

  programs.clang-format.enable = true;
  programs.clang-format.package = pkgs.llvmPackages_23.clang-tools;
  programs.cmake-format.enable = true;
  programs.cmake-format.includes = [ "*CMakeLists.txt" ];
  programs.nixfmt.enable = true;
}
