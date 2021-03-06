# cran-comments.md


                
## References
Added reference to SensAnalysisMLP function where the formulation to calculate the partial derivatives is explained.
I am currently writing a paper for the Journal of Statistic Software about the package.

## donttest{} examples

The `SensAnalysisMLP.R` function contains `\donttest{}` examples which produce animations that take >5sec that users need to know about, but cause issues in examples and checks. 
The manipulations before the animation rending is already tested in the example which is not wrapped by `\donttest{}`.


## Test environments
* Windows 10 Home x64, R 3.6.0
* ubuntu 16.04.6 (on travis-ci), R (oldrel, release and devel)
* Mac OS X 10.13.6 (on travis-ci), R (oldrel, release and devel)
* win-builder, R (oldrelease, release and devel)

### R CMD check

-- R CMD check results ----------------------------------- NeuralSens 0.2.2 ----
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

### Comments from the reviewer
* \dontrun examples has been changed to \donttest because it can be run if a neural network model is trained, but can not be checked automatically.
* Commented code lines of examples has been erased.
* *immediate* call of on.exit() has been added when calling to par()
* Use of <<- has been changed to avoid changing the global environment
* installed.packages() has been changed to requireNamespace()
* Errors in examples when running check with --run-donttest (the reason why the package was archived) have been corrected
* Regarding a reference about the method, there is not a published article yet that can be provided as reference. An article about the NeuralSens package is in minor revision by the Journal of Statistical Software.

