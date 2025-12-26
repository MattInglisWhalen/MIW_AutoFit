# MIW's AutoFit -- Dev Notes

*A reminder for future Matt on how to handle linting / styling / testing*

## Testing

For a simple evaluation of the test suite, in /MIW_AutoFit type

> pytest

For a coverage report in easy-to-read html, instead use

> pytest --cov-report=html

## Linting

In order to see how well the codebase is adhering to best practices and/or coding guidelines,
in /MIW_AutoFit/autofit type

> pylint --rcfile=pylintrc/.pylintrc .

The rcfile basically just drops the requirement that **all** variables adhere to strict 
snake_case naming, which is annoying for things like `A` in a templated utility function,
or `HQIC` which is an acronym, or `error_L`/`error_R` when the lowercase . This option is located in .pylintrc at line 424
(the disable= line). 

If you want to use the default settings, just type

> pylint .

Careful! The default pylint line length is 88 but this project uses 100.

If you'd rather focus on a single lint-error type, use e.g.

> pylint --disable=all --enable=invalid-name .

To ignore all instances of bad design, rather than just bad code style, use

>   pylint --rcfile=pylintrc/.pylintrc_bad_design .


## Code Style

To check the differences between the current code and what python black would suggest, 
in /MIW_AutoFit type

> black --check --diff .

To have these
changes implemented automatically, just type

> black .