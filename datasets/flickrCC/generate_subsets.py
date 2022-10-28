import random
import os
import gzip

MAX_PBUFFER_SIZE = 2**20  # 1MB
BUFFER_SIZE = 2**7


class Bucket:
    def __init__(self, filename):
        self.filename = filename
        self.dirname = os.path.dirname(self.filename)
        os.makedirs(self.dirname, exist_ok=True)

        self.buffer = []
        self.pbuffer = filename.replace('.tsv.gz', '.buffer.tsv')


    def load_pbuff(self):
        contents = []
        if os.path.isfile(self.pbuffer):
            contents = list(open(self.pbuffer, 'rb'))
        return contents

    def append_pbuff(self, contents):
        with open(self.pbuffer, 'ab') as fp:
            fp.write(b''.join(contents))

    def reset_pbuff(self):
        open(self.pbuffer, 'w').close()

    def clean_pbuff(self):
        os.remove(self.pbuffer)

    def load_gzip(self):
        contents = []
        if os.path.isfile(self.filename):
            contents = list(gzip.open(self.filename, 'rb'))
        return contents

    def flush_all_buffers(self):
        contents = self.load_pbuff() + self.buffer
        self.buffer = []
        self.reset_pbuff()
        if not contents:
            return

        with gzip.open(self.filename, 'ab') as fp:
            fp.write(b"".join(contents))

    def add_item(self, x):
        self.buffer += [x]
        if len(self.buffer) == BUFFER_SIZE:
            self.commit_pbuff()

    def commit_pbuff(self):
        # Commit to a persistent buffer
        self.append_pbuff(self.buffer)
        self.buffer = []

        # Check pbuff size in MB
        pbuff_size = os.path.getsize(self.pbuffer)
        if pbuff_size > MAX_PBUFFER_SIZE:
            # Commit to compressed bucket
            self.flush_all_buffers()

    def commit(self):
        self.flush_all_buffers()


class Storage:
    def __init__(self, root):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.buckets = {}

    def commit(self):
        for bk in self.buckets.keys():
            self.buckets[bk].commit()

    def commit_pbuffs(self):
        for bk in self.buckets.keys():
            self.buckets[bk].commit_pbuff()

    def clean(self):
        for bk in self.buckets.keys():
            self.buckets[bk].clean_pbuff()

    def add_item(self, x):
        try:
            bucket = x.decode('utf-8').strip().split('\t')[-1]
            bucket = f"{int(bucket):04d}"
        except ValueError:
            # If bucket has not been assigned
            bucket = random.randint(0, 10000-1)
            bucket = f"{bucket:04d}"
            x_list = x.decode('utf-8').strip().split('\t')
            x_list.append(bucket)
            x = ('\t'.join(x_list)+'\n').encode('utf-8')

        if bucket not in self.buckets:
            self.buckets[bucket] = Bucket(f"{self.root}/{bucket[0]}/{bucket[1]}/{bucket[2]}/{bucket}.tsv.gz")
        self.buckets[bucket].add_item(x)

    def add_chunk(self, chunk):
        for x in chunk:
            self.add_item(x)


def process_chunk(path, output_dir, gzipped=False):
    all_files = []
    for root, subdirs, files in os.walk(path):
        if gzipped:
            all_files += [f"{root}/{fn}" for fn in files if fn.endswith('.tsv.gz')]
        else:
            all_files += [f"{root}/{fn}" for fn in files if fn.endswith('.tsv')]

    db = Storage(output_dir)

    for fn in all_files:
        if gzipped:
            data = list(gzip.open(fn, 'rb'))
        else:
            data = list(open(fn, 'rb'))
        db.add_chunk(data)
    db.commit()
    db.clean()


def split_into_buckets(input_dir, output_dir):
    all_dirs = sorted([path for path, subdirs, files in os.walk(input_dir)
                       if len([fn for fn in files if fn.endswith('.tsv.gz')])])
    for d in all_dirs:
        print(d)
        process_chunk(d, output_dir=output_dir, gzipped=True)


def parse_metadata(index_path):
    default_bucket = index_path.split('/')[-1].split('.')[0]

    with gzip.open(index_path, 'rb') as fp:
        for x in fp:
            x_list = x.decode('utf-8').strip().split('\t')
            flickr_iid = x_list[0]
            image_url = x_list[2]
            bucket = x_list[-1]
            try:
                int(bucket)
            except ValueError:
                bucket = default_bucket
            local_file = f"images/{bucket[0]}/{bucket[1]}/{bucket[2]}/{bucket}/{image_url.split('/')[-1]}"

            yield flickr_iid, image_url, local_file


def create_random_image_list(list_fn, image_dir, bucket_start=0, bucket_end=10000, max_images=None):
    files = [f"{image_dir}/{f'{d:04d}'[0]}/{f'{d:04d}'[1]}/{f'{d:04d}'[2]}/{f'{d:04d}'}.tsv.gz"
             for d in range(bucket_start, bucket_end)]
    files = [fn for fn in files if os.path.isfile(fn)]

    os.makedirs(os.path.dirname(list_fn), exist_ok=True)
    with gzip.open(list_fn, 'wb') as fp:
        n_images = 0
        for bucket_filename in files:
            bucket_content = list(set(parse_metadata(bucket_filename)))
            random.shuffle(bucket_content)
            for image_iid, image_url, image_fn in bucket_content:
                fp.write(('\t'.join([image_iid, image_url, image_fn])+'\n').encode('utf-8'))
                n_images += 1
                if max_images is not None and n_images >= max_images:
                    return
                if n_images % 100000 == 0:
                    print(f"Added {n_images//1000000}M images to {list_fn}. (Current bucket: {bucket_filename})")


def generate_subsets(full_list_fn):
    subset_size_millions = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    subset_files = {ss: gzip.open(full_list_fn.replace('.tsv.gz', f'-{ss}M.tsv.gz'), 'wb')
                    for ss in subset_size_millions}
    with gzip.open(full_list_fn, 'rb') as fp:
        n_images = 0
        while True:
            x = fp.readline()
            if not x: break
            n_images += 1
            if n_images % 100000 == 0:
                print(n_images)
            for ss in subset_size_millions:
                if n_images <= int(ss*1000000):
                    subset_files[ss].write(x)

    for ss in subset_size_millions:
        subset_files[ss].close()
        if n_images <= int(ss*1000000):
            os.remove(full_list_fn.replace('.tsv.gz', f'-{ss}M.tsv.gz'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl_folder', default='meta_by_date', help='Subset file.')
    parser.add_argument('--images_dir', default='images', help='Subset file.')
    parser.add_argument('--lists_dir', default='lists', help='Subset file.')
    args = parser.parse_args()

    split_into_buckets(input_dir=args.crawl_folder, output_dir=args.images_dir)
    create_random_image_list(f'{args.lists_dir}/flickr.tsv.gz', args.images_dir)
    generate_subsets(f'{args.lists_dir}/flickr.tsv.gz')


if __name__ == '__main__':
    main()

