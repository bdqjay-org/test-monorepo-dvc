#!/bin/bash
set -e
echo ""
echo "Hey, I will automate the following:"
echo -e "\t1. Creation of a new ML project with uv"
echo -e "\t2. Generation of the folder structure"
echo -e "\t3. Installation of dependencies"
echo -e "\t4. Setting up and installation of pre-commit hooks"
echo -e "\t5. Addition of the project to PYTHONPATH"
echo "***************************************************************"
echo "I will need your input for the following (and even later!):"
echo ""
read -p "What would you like to name your project?    " project_name
echo ""
echo "Thank you! The project setup will begin now."
echo ""
echo "################################################################"
echo "Creating the project titled '$project_name' with uv ..."
echo ""
uv init $project_name
cd $project_name
rm -f main.py
echo ""
echo "Project creation done."
echo "################################################################"
echo "Creating the folder structure for the project ..."
mkdir data models notebooks reports src tests
cd data
mkdir external interim processed raw
cd ..
cd src
mkdir data evaluate features report stages train utils
touch __init__.py
cd ..
echo ""
echo "Folder structure generation done."
echo "################################################################"
echo "Installing the necessary dependencies ..."
echo ""
uv add -r ../requirements.txt
echo ""
echo "Dependencies installation done."
echo "################################################################"
echo ""
echo "Configuring 'shared' as a local dependency ..."
awk '
  BEGIN { added=0 }
  /^\s*\]/ && added==0 {
    print "    \"shared\","
    added=1
  }
  { print }
' pyproject.toml > tmp.toml && mv tmp.toml pyproject.toml


cat <<EOF >> pyproject.toml

[tool.uv.sources]
shared = { path = "../shared", editable = true }
EOF
echo "Dependency linking complete."
echo "################################################################"
echo "Setting up pre-commit hooks ..."
echo ""
cat <<EOF > .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black-pre-commit-mirror
    rev: 25.1.0
    hooks:
      - id: black
        language_version: python3.10
  - repo: https://github.com/PyCQA/flake8
    rev: 7.2.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.16.0
    hooks:
      - id: mypy
        args:
          [
            --disallow-untyped-defs,
            --disallow-incomplete-defs,
            --disallow-untyped-calls,
            --ignore-missing-imports,
          ]
        additional_dependencies: [types-requests, types-PyYAML]
  - repo: https://github.com/asottile/reorder_python_imports
    rev: v3.15.0
    hooks:
      - id: reorder-python-imports
  - repo: https://github.com/astral-sh/uv-pre-commit
    rev: 0.7.10
    hooks:
      - id: uv-lock
  - repo: https://github.com/iterative/dvc
    rev: main
    hooks:
      - id: dvc-pre-commit
        additional_dependencies: [".[all]"]
        language_version: python3
        stages:
          - commit
      - id: dvc-pre-push
        additional_dependencies: [".[all]"]
        language_version: python3
        stages:
          - push
      - id: dvc-post-checkout
        additional_dependencies: [".[all]"]
        language_version: python3
        stages:
          - post-checkout
        always_run: true
EOF
uvx pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
echo ""
echo "Pre-commit installed and initialized."
echo "################################################################"
echo ""
echo "Adding the project to PYTHONPATH ..."
project_dir=$PWD
echo "export PYTHONPATH="${PYTHONPATH}:$project_dir"" >> ~/.bashrc
echo "export PYTHONPATH="${PYTHONPATH}:$project_dir"" >> ~/.profile
source ~/.bashrc
echo ""
echo "Environment and folder structure setup complete!"
echo ""
echo "Happy building!" 
echo "################################################################"