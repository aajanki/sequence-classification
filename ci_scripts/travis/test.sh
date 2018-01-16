set -e

if [[ "$COVERAGE" == "true" ]]; then
    coverage run --source=$MODULE setup.py test
else
    python setup.py test
fi
