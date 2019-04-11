#!/usr/bin/env sh

if [ -x "$(command -v kaggle)" ]; then
  printf "Downloading data to: %s\\n" "$1"
  mkdir -p "$1"
  cd "$1" &&
    kaggle competitions download -c ujnn2019-1 &&
    cd - || exit 1
  printf "Script ran successfully"
else
  printf "Unable to download data, 'kaggle' command not found.\\n"
  printf "Install Kaggle official command line API (https://github.com/Kaggle/kaggle-api).\\n"
  exit 1
fi
