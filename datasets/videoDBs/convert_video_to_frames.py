import av
import tqdm
import os
import multiprocessing as mp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/grogu/user/pmorgado/datasets/krishnacam/videos', help='Root directory of video dataset.')
    parser.add_argument('--frames_dir', default='/grogu/user/pmorgado/datasets/krishnacam/frames', help='Root directory to place frames.')
    parser.add_argument('--workers', default=0, type=int, help='Number of parallel workers.')
    args = parser.parse_args()

    videos_fns = [f"{root}/{fn}" for root, subdir, files in os.walk(args.video_dir)
                  for fn in files if fn.endswith('.mp4') or fn.endswith('.avi')]
    jobs = []
    for vfn in videos_fns:
        basename = os.path.basename(vfn).split('.')[0]
        dst_fns = f'{args.frames_dir}/{basename}/%05d.jpg'
        os.makedirs(os.path.dirname(dst_fns), exist_ok=True)

        # Figure out how much to downscale
        ctr = av.open(vfn)
        h = ctr.streams.video[0].codec_context.coded_height
        w = ctr.streams.video[0].codec_context.coded_width
        scale = '' if min(w, h) < 256 else '-vf scale="-2:\'min(256,ih)\'"' if w > h else 'scale="\'min(256,iw)\':-2"'
        ctr.close()

        # Figure out how much to downscale
        command = f'ffmpeg -y -i {vfn} {scale} -r 10 {dst_fns}'
        jobs += [command]
        print(command)

    pool = mp.Pool(args.workers)
    for _ in tqdm.tqdm(pool.imap_unordered(os.system, jobs), total=len(jobs)):
        pass



if __name__ == '__main__':
    main()
