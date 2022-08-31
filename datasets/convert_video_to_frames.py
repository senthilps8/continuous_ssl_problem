import glob
import av
import tqdm
import os
import multiprocessing as mp

VIDEOS_PATH = '/grogu/user/pmorgado/datasets/krishnacam/videos'
FRAMES_PATH = '/grogu/user/pmorgado/datasets/krishnacam/frames'

videos_fns = glob.glob(f"{VIDEOS_PATH}/*")
jobs = []
for vfn in videos_fns:
    basename = vfn.split('/')[-1].split('.')[0]

    ctr = av.open(vfn)
    h = ctr.streams.video[0].codec_context.coded_height
    w = ctr.streams.video[0].codec_context.coded_width
    scale = '' if min(w, h) < 256 else '-vf scale="-2:\'min(256,ih)\'"' if w > h else 'scale="\'min(256,iw)\':-2"'
    ctr.close()

    command = f'mkdir -p {FRAMES_PATH}/{basename} && ffmpeg -y -i {vfn} {scale} -r 10 {FRAMES_PATH}/{basename}/%05d.jpg'
    jobs += [command]
    print(command)


pool = mp.Pool(10)
for _ in tqdm.tqdm(pool.imap_unordered(os.system, jobs), total=len(jobs)):
    pass
