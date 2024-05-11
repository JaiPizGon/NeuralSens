# cran-comments.md

## Feature
Significance test uses bootstrap with the call of the `caret::train()` function. However, this is very sensitive to changes in how the model is trained, giving errors when some necessary variable is not correctly stored in the global environment (due to calling the `caret::train()` function inside another function, for example). Therefore, necessary variables can now be passed as extra arguments to the `SensAnalysisMLP()` function.

## donttest{} examples

The `SensAnalysisMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. The `HessianMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. 
The manipulations before the animation rending is already tested in the examples which are not wrapped by `\donttest{}`.

## Test environments
* Windows 10 Home x64, R 4.3.2
* Ubuntu Linux 22.04.1 LTS, R-release, GCC (r-hub)
* Fedora Linux, R-devel, clang, gfortran (r-hub)
* win-builder, R (oldrelease, release and devel)

### R CMD check
── R CMD check results ─────── NeuralSens 1.1.1 ────
Duration: 6m 41.6s

0 errors ✔ | 0 warnings ✔ | 0 notes ✔

R CMD check succeeded

## win-builder
- oldrelease pass
- release pass
- devel pass


