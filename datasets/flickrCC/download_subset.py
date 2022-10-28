import sys
import os
import gzip
import multiprocessing as mp

import urllib.request
import urllib.error

from io import BytesIO
from PIL import Image

sys.path.insert(0, 'openverse-catalog/openverse_catalog/dags')
from common.requester import DelayedRequester

RESIZE = 256


def download_and_resize(req, url, out_fn, size=None):
    if size is None:
        req.get(url, out_fn)
    else:
        img = Image.open(BytesIO(req.get(url).content))
        w, h = img.size
        factor = size / min(w, h)
        img = img.resize((int(w * factor), int(h * factor)))
        img.save(out_fn)


def download_chunk(image_chunk, root='.'):
    req = DelayedRequester(delay=0.1)
    for idx, image_data in enumerate(image_chunk):
        if idx % 250 == 0:
            print(f'{idx}/{len(image_chunk)} images downloaded!')
        image_d, image_url, image_fn = image_data.decode('utf-8').strip().split('\t')
        image_fn = f"{root}/{image_fn}"

        # Skip images that have been already downloaded
        if os.path.isfile(image_fn):
            continue

        try:
            # Download image and resize
            os.makedirs(os.path.dirname(image_fn), exist_ok=True)
            download_and_resize(req, image_url, image_fn, RESIZE)
        except urllib.error.HTTPError as e:
            # Skip if url is broken
            continue
        except KeyboardInterrupt:
            if os.path.isfile(image_fn):
                os.remove(image_fn)
            raise KeyboardInterrupt


def download_images(root_dir, image_list_fn, workers=0):
    images = list(gzip.open(image_list_fn, 'r'))
    chunk_size = len(images)//max(workers, 1)
    image_chunks = [images[i*chunk_size:(i+1)*chunk_size] for i in range(max(workers, 1))]
    from functools import partial
    download_fcn = partial(download_chunk, root=root_dir)

    if workers == 0:
        for chunk in image_chunks:
            download_fcn(chunk)
    else:
        pool = mp.Pool(workers)
        pool.map(download_fcn, image_chunks)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./', help='Root directory to place downloaded images.')
    parser.add_argument('--subset', default='lists/flickr.tsv.gz', help='Subset file.')
    parser.add_argument('--workers', default=0, type=int, help='Number of parallel workers.')
    args = parser.parse_args()

    download_images(args.root_dir, args.subset, workers=args.workers)


if __name__ == '__main__':
    main()
