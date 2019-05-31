# cran-comments.md

## donttest{} examples

The `SensAnalysisMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. 
The manipulations before the animation rending is already tested in the example which is not wrapped by `\donttest{}`.


## Test environments
* Windows 10 Home x64, R 3.6.0
* ubuntu 14.05.5 (on travis-ci), R (oldrel, release and devel)
* Mac OS X 10.13.3 (on travis-ci), R (oldrel, release and devel)
* win-builder, R (oldrelease, release and devel)
* macOS Sierra 10.12.6, R 3.5.1

### R CMD check

-- R CMD check results ----------------------------------- NeuralSens 0.0.2 ----
Duration: 27.5s

0 errors √ | 0 warnings √ | 0 notes √


## Travis CI
- linux
    - oldrel pass
    - release pass
    - devel pass
- osx
    - oldrel pass
    - release pass
    - devel ERROR:
   Installing R
   292.31s$ brew update >/dev/null
   4.05s$ curl -fLo /tmp/R.pkg https://r.research.att.com/el-capitan/R-devel/R-devel-el-capitan-signed.pkg
   curl: (22) The requested URL returned error: 404 Not Found
   The command "eval curl -fLo /tmp/R.pkg https://r.research.att.com/el-capitan/R-devel/R-devel-el-capitan-signed.pkg " failed. Retrying, 2 of 3.

## win-builder
- oldrelease pass
- release pass
- devel pass