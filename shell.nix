# shell.nix
{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python313.buildEnv.override {
    extraLibs = with pkgs.python39Packages; [
      numpy
      pandas
      requests
      
    ];
  };
in
pkgs.mkShell {
  buildInputs = [ pythonEnv ];
}
