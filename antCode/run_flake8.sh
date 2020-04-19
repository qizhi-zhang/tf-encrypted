#!/bin/bash

pip install -i https://pypi.antfin-inc.com/simple/ antflake8

export LC_ALL=zh_CN.utf-8

changed_pyfiles=$(git diff --name-only HEAD origin/master | { grep ".py$" || test $? = 1; })
if [ -z "$changed_pyfiles" ]; then changed_pyfiles=$(find . -type f -name '*.py'); fi

echo "Lint Files:"
echo $changed_pyfiles
echo

# Run the linter.
err_state=0
for f in $changed_pyfiles; do
    if [[ $f == *"__init__.py"* ]]; then
        continue
    fi
    echo "Linting ${f}"
    PYTHONPATH=${PYTHONPATH}:$(dirname $f) stdbuf -oL antflake8 $f --show-source
    err_state=$((err_state + $?))
    echo
done

[ $err_state -eq 0 ] || exit 1
echo "Lint Finished"