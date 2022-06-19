# cran-comments.md


## donttest{} examples

The `SensAnalysisMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. The `HessianMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. 
The manipulations before the animation rending is already tested in the examples which are not wrapped by `\donttest{}`.

## Archived package
NeuralSens was archived due to an error in some examples due to obsolete `neural` package. 
A Note arises in r-hub testing environments due to archived package:

❯ checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Jaime Pizarroso Gonzalo <jpizarroso@comillas.edu>'
  
  New submission
  
  Package was archived on CRAN

## Test environments
* Windows 10 Home x64, R 4.2.0
* Ubuntu Linux 20.04.1 LTS, R-release, GCC (r-hub)
* Fedora Linux, R-devel, clang, gfortran (r-hub)
* win-builder, R (oldrelease, release and devel)

### R CMD check
── R CMD check results ─────── NeuralSens 1.0.1 ────
Duration: 5m 29s

0 errors ✔ | 0 warnings ✔ | 0 notes ✔

R CMD check succeeded

## win-builder
- oldrelease pass
- release pass
- devel pass


