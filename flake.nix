# FILENAME: flake.nix
{
  description = "theta-hyper-shared-ptr: 8192-CPU rseq std::shared_ptr ABI-compatible and handle-tagged concurrent smart pointers";

  inputs = {
    theta-utils.url = "github:LoganEvans/theta-utils";
    nixpkgs.follows = "theta-utils/nixpkgs";
    flake-utils.follows = "theta-utils/flake-utils";
    treefmt-nix.follows = "theta-utils/treefmt-nix";
  };

  outputs =
    {
      self,
      theta-utils,
      ...
    }:
    theta-utils.lib.mkThetaFlake {
      inherit self;
      pname = "theta-hyper-shared-ptr";
      packageFn = ./package.nix;
      treefmtModule = ./treefmt.nix;
    };
}
