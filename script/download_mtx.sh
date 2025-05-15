#!/bin/bash

mtx_dir="./mtx"

links=(
  "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt200.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/Koutsovasilis/F1.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/sx-stackoverflow.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/inline_1.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian19.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Bump_2911.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_1.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology1.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/HB/nos2.tar.gz"
  # "https://suitesparse-collection-website.herokuapp.com/MM/HB/fs_541_3.tar.gz"
)

for link in "${links[@]}"; do
  filename=$(basename "$link")
  foldername="${filename%%.*}"
  
  if [ ! -f "$filename" ]; then
    echo "Downloading $filename..."
    wget "$link"
    if [ $? -ne 0 ]; then
      echo "Failed to download $filename"
      exit 1
    fi
  else
    echo "$filename already exists, skipping download."
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
    mv "$foldername"/* "$mtx_dir"/
    rm -rf "$foldername"
  else
    echo "No files to move from $foldername."
    exit 1
  fi
done