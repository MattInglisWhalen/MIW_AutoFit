[![Tests](https://github.com/MattInglisWhalen/MIW_AutoFit/actions/workflows/tests.yml/badge.svg)](https://github.com/MattInglisWhalen/MIW_AutoFit/actions/workflows/tests.yml)
   ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_windows.json) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_macos.json) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MattInglisWhalen/4fb351291438ee5d4f772ff9966f06d3/raw/covbadge_ubuntu.json)
# MIW's AutoFit
 
!["MIW's AutoFit splash image"](/autofit/images/splash.png "Splash image")

 Have data but don't know how to fit it? This tool will automatically 
 tell you the best function to model your data, along with the best-fit parameters and uncertainties.

---

### Executables / Apps 

Windows: [Version 0.3⭳](https://ingliswhalen.files.wordpress.com/2023/11/d6098-miw_autofit_03.zip)

MacOSX: [Version 0.3⭳](https://ingliswhalen.files.wordpress.com/2023/11/dc42b-miw_autofit_03_osx.zip)

Ubuntu: [Version 0.3⭳](https://ingliswhalen.files.wordpress.com/2023/11/48ae5-miw_autofit_03_linux.zip)

### Tutorials

See the [tutorials pages here](https://mattingliswhalen.github.io/MIWs_AutoFit_Tutorial_1/).

## Outline

MIW's AutoFit  is the result of pigheadedly generating as many functional models as you'll let it, 
then ranking each one based on its goodness of fit.

A model in MIW's AutoFit is a treelike structure where each node represents a primitive functions 
(like $\cos(x)$ or $x^2$), and the relations between each node can indicate composition, summation,
or multiplication. A 'parent' node can have 'children' nodes, so that the output $f(x)$ of 
evaluating the parent at a particular input $x$ is given by 

$$f(x) = \mathrm{parent}(\mathrm{child}_1(x)+\mathrm{child}_2(x)+\ldots)$$

Therefore, parents compose with the sum of their children, where each child represents 
one term in the summation.
Meanwhile, each 'child' actually represents a set of 'brothers', so that 

$$\mathrm{child}(x) = \mathrm{brother}_1(x)\cdot\mathrm{brother}_2(x)\cdot\ldots$$

In this way, multiplications are implemented in each term of a summation. 

!["MIW's AutoFit Tree Structure"](/autofit/images/hierarchy.png "Hierarchy image")

For example, the above image represented the model

$$f(x) = A\cos(Bx+C) \cdot \sin(D\exp(x)\log(x)+Ex^2)$$

## Contributions

MIW's AutoFit is no longer being actively developed, but engagement through comments and suggestions are
likely to renew my interest in this project.



