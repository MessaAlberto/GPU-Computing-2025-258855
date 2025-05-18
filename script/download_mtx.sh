#!/bin/bash

mtx_dir="./mtx"

links=(
  "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_1.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology1.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Koutsovasilis/F1.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/inline_1.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt200.tar.gz"
  "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/sx-stackoverflow.tar.gz"
)

for link in "${links[@]}"; do
  filename=$(basename "$link")
  foldername="${filename%%.*}"
  expected_mtx="${foldername}.mtx"

  echo "$filename"
  echo "$foldername"
  echo "$expected_mtx"
  
  if [ ! -f "$mtx_dir/$expected_mtx" ]; then
    echo "Downloading $filename..."
    wget "$link"
    if [ $? -ne 0 ]; then
      echo "Failed to download $filename"
      exit 1
    fi
  else
    echo "$mtx_dir/$expected_mtx already exists, skipping download."
    continue
  fi

  # Decompress the file
  if [[ "$filename" == *.tar.gz ]]; then
    echo "Decompressing $filename..."
    gzip -d "$filename"
    if [ $? -ne 0 ]; then
      echo "Failed to decompress $filename"
      exit 1
    fi
    filename="${filename%.gz}"
  fi

  # Extract the tar file
  if [[ "$filename" == *.tar ]]; then
    echo "Extracting $filename..."
    tar -xf "$filename"
    if [ $? -ne 0 ]; then
      echo "Failed to extract $filename"
      exit 1
    fi
    rm "$filename"
  fi

  # Move the extracted files to the mtx directory
  if [ -d "$foldername" ]; then
    echo "Moving files to $mtx_dir..."
    mkdir -p "$mtx_dir"
    mv "$foldername/$foldername.mtx" "$mtx_dir"/
    rm -rf "$foldername"
  else
    echo "No files to move from $foldername."
    exit 1
  fi
done