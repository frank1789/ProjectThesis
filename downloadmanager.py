#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token, }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024 * 1024
    file_size = int(response.headers.get('Content-Length', 1))
    # file_size = int(response.headers.get('content-length'))
    num_bars = int(file_size / CHUNK_SIZE)
    pbar = tqdm(
        response.iter_content(chunk_size=CHUNK_SIZE), total=num_bars, unit='KB', desc=destination
    )
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()
    print("\n")


if __name__ == "__main__":
    file_id = "16ZGOwfLrN_54eR5yhXFtp8tmJ10Abudr"
    destination = os.path.join(os.getcwd(), "dataset.zip")
    if not os.path.exists(destination):
        print("Download file:")
        download_file_from_google_drive(file_id, destination)
    else:
        print("File exist")
        exit(0)
