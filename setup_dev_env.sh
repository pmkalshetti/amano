#!/bin/sh

# you need to be in the project's root directory when sourcing this bash script
# use `source filename.sh` instead of `bash filename.sh`
# because `conda activate xxx` fails as the functions are not available in subshell (Ref: # https://github.com/conda/conda/issues/7980#issuecomment-441358406)
conda activate env_amano
path_src=`realpath src`
export PYTHONPATH=$path_src