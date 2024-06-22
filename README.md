[![Tests](https://github.com/MattInglisWhalen/MIW_AutoFit/actions/workflows/tests.yml/badge.svg)](https://github.com/MattInglisWhalen/MIW_AutoFit/actions/workflows/tests.yml)
   ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_windows.json) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_macos.json) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_ubuntu.json)
# MIW's AutoFit
 
!["MIW's AutoFit splash image"](/autofit/images/splash.png "Splash image")

 Have data but don't know how to fit it? This tool will automatically 
 tell you the best function to model your data, along with the best-fit parameters and uncertainties.

---

### Executables / Apps 

Windows: [Version 0.4⭳](https://dl.dropboxusercontent.com/scl/fi/fv1x2h01qfl4sylskqzzu/MIWs_AutoFit_04_win.zip?rlkey=z3rgph8rgz4wpdx0jimrklq5i&dl=0)

MacOSX: [Version 0.3⭳](https://ingliswhalen.files.wordpress.com/2023/11/a7ead-miw_autofit_free_03_osx.zip)

Ubuntu: [Version 0.3⭳](https://ingliswhalen.files.wordpress.com/2023/11/5f4e1-miw_autofit_free_03_linux.zip)

### Tutorials

See the [tutorials pages here](https://mattingliswhalen.github.io/MIWs_AutoFit_Tutorial_1/).

## Representation of Functions

MIW's AutoFit is the result of pigheadedly generating as many functional models as you'll let it, 
fitting eacg model to the input data, then ranking each one based on its goodness of fit.

A functional model in MIW's AutoFit is represented as a treelike structure where each node represents a primitive function 
( like $\cos(x)$ or $x^2$ ), and the relationship between each node indicates whether
the primitives are related through composition, summation, or multiplication. A `parent` node $P$ with primitive function
$p(x)$ can have `child` nodes $C_1 \ldots C_N$ that all have their own respective primitives $c_1(x) \ldots c_N(x)$. 
The output $P(x)$ of evaluating the parent at a particular input $x$ is given by 

$$P(x;C_1 \ldots C_N) = P(C_1(x) + C_2(x) + \ldots + C_N(x))$$

Therefore, parents compose with the sum of their children, where each child represents one term in the summation.

Meanwhile, each `child` term represents the multiplication of a set of `brother` nodes
$B_1 \ldots B_M(x)$ with primitive function $b_1(x) \ldots b_M(x)$, so that 

$$C(x; B_1 \ldots B_M) = B_1(x) \cdot B_2(x) \cdot \ldots \cdot B_M(x)$$

In this way multiplicative relationships are implemented in each term of a summation.

!["MIW's AutoFit Tree Structure"](/autofit/images/hierarchy.jpg "Hierarchy image")

For example, the above image represents the model

$$f(x) = A\cos(Bx+C) \cdot \sin(D\exp(x)\cdot\log(x)+Ex^2) + \frac{F}{x}$$

## Goodness of Fit

As a trained physicist, my favourite measure for the goodness of fit is called the reduced chi-squared statistic

$$\chi^2_{red} = \frac{\chi^2}{N-d} = \frac{1}{N-d} \sum_{i=1}^{N} \left(\frac{y_i-f(x_i)}{\sigma_i}\right)^2$$

where the sum is over each of the $N$ datapoints and $d$ is the number of degrees of freedom in
the model (i.e. $d$ is the number of parameters to be fit). I like this statistic because when
the variations $\varepsilon_i = y_i-f(x_i)$ are normally distributed with standard deviation $\sigma_i$
then a good model should have $\chi^2_{red}$ close to 1. If a model produces a reduced $\chi^2$ much more than 1
then the model doesn't fit very well (or the uncertainties in the data are underestimated) and if the model
produces a reduced $\chi^2$ much less than 1 then the model is likely overfitting the data
(or the uncertainties in the data are overestimated).

The reduced  $\chi^2$ is closely related to another test statistic that is often used as a goodness-of-fit
criterion, called Akaike's Information Criterion. AIC is defined as 

$$AIC = 2d + \chi^2 + \mathrm{const}$$

where the constant term depends only on the data rather than the model. This criterion is often used in
order to get an estimate of relative probabilities of correctness when comparing models; if the top model (lowest AIC)
has $AIC_\mathrm{top}$ then the probability that an alternative model with $AIC_\mathrm{alt}$ would better fit the data is given by
$$P_\mathrm{alt} = \exp\left(\frac{AIC_0-AIC_\mathrm{alt}}{2}\right)$$

There are many other alternatives to the reduced $\chi^2$ or AIC. MIW's AutoFit has also implemented the corrected
Akaike's Informaton Criterion for small datasets, and the Bayes Information Criterion, and the Hannan-Quinn Information Criterion.
Simply toggle the option to use your preferred choice!

## Initial Parameter Estimation

One of the most difficult problems in fitting a function is finding an initial set of parameters
that allows the fit algorithm to find a global minimum. Without a good set of seed parameters, a multivariate
minimization algorithm can easily become stuck in a local minimum. 

A concept that is hammered into physics students early on in their studies is that of dimensional scaling.
If a dataset is built to capture all the relevant behaviour of a function, then the set of x-values chosen
in the dataset are presumably relevant to any model that properly describes that dataset. A 
`canonical` value \[X\] for x might then be the average of all the x-values in the dataset. Similarly, over 
that same domain of x-values, the measured values making up the range of y-values are also relevant to the model.
So a `canonical` value \[Y\] for y might be the average of all the y-values.

Applying this intuition to the model is simply a matter of determining how a particular parameter *scales* with
repect to x or y. In a simple model like a linear equation $y=mx+b$, the parameter $m$ scales like $[Y]/[X]$ and
the parameter $b$ scales like $[Y]$. So we might first estimate $m$ to be mean($y$)/mean($x$) and $b$ as mean($y$).

MIW's AutoFit recursively applies this behaviour to get the initial guess for how all parameters in any model
should scale. It knows that in the model $y=A\exp(Bx)$, $A$ should scale like \[Y\] and $B$ should scale like 1/\[X\]. It knows that
in the model $y=A\cos(Bx+C/x)$, $A$ should scale like \[Y\], and $B$ should scale like 1/\[X\], and C should scale like \[X\].

But it's not as simple as taking the mean of x- and y-values. Sometimes the *extent* of x- and y-values is more important than
their mean, so the difference between the maximum and minimum values is used instead. And for oscillatory behaviour, 
it's the fundamental period of the oscillations that's more useful as a canonical length measurement. MIW's AutoFit
can detect these scenarios, and intelligently implements different behaviours depending on the form of the model.


## Contributions

MIW's AutoFit is no longer being actively developed, but engagement through comments and suggestions are
likely to renew my interest in this project.



