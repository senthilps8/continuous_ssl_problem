import numpy as np

MAX_LEN = 120
FOLDER = f'/grogu/user/pmorgado/datasets/flickrDB/'
if __name__ == '__main__':
    for DB_SZ in [1, 2, 5, 10, 20, 50, 100, 200]:
        mmap = np.memmap(f"{FOLDER}/filelist_{DB_SZ}M.memmap",
                         dtype=f"S{MAX_LEN}",
                         mode="w+",
                         shape=(int(DB_SZ*1e6),))
        with open(f"{FOLDER}/filelist_{DB_SZ}M", 'r') as f:
            for i, ln in enumerate(f):
                if i == len(mmap):
                    break
                fn = ln.split()[0]
                assert len(fn) < MAX_LEN
                fn + '\n' + ' ' * (MAX_LEN - 1 - len(fn))
                mmap[i] = fn
            mmap.flush()
