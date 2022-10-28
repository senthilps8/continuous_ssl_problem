# Data Preparation
This page provides instructions for preparing the various datasets used in the paper. 

### Environment
We used openverse-catalog to crawl flickr data. To run this pipeline (both crawling and image downloading code), please set up your python environment as follows.
```
cd datasets/flickrCC
conda create -n flickrCC python=3.10
conda activate flickrCC

git clone https://github.com/WordPress/openverse-catalog.git
pip install -r openverse-catalog/requirements_tooling.txt
pip install lxml apache-airflow tldextract requests-file requests-oauthlib retry pillow numpy tqdm av
```

# Flick-CC
We provide code for collecting flickr data and/or downloading the exact image subsets used in the paper.

### Crawl flickr and generate datasets
This step is not required (pre-extracted subsets can be downloaded below).
This code can be used to collect all CC images in Flickr between any two dates.
It will produce a true uncurated image dataset, as no filters are applied.
The subsets used in the paper are based on image data between 2004 and 2021.
```
cd datasets/flickrCC
python crawl_flickr.py --init_date 2004-01-01 --end_date 2022-01-01 --crawl_folder meta --workers 8
python generate_subsets.py --crawl_folder meta --images_dir images --lists_dir ../lists
```

### Get subsets from the paper
The subsets used in the paper are available for download.
Download one of the subsets below, and place it under `datasets/lists`.

|     Subset     | File Size |                MD5               |                                            Link                                            |
|:--------------:|:---------:|:--------------------------------:|:------------------------------------------------------------------------------------------:|
|  Flickr-100k   |   2.3Mb   | 2da381ffaba9b37f417b06fdc807e426 | [link](https://drive.google.com/file/d/1M1NAxkDvlf0Lpxvq2FqrgNSeTMz0JL2Y/view?usp=sharing) |
|  Flickr-200k   |   4.5Mb   | d12a135397ae35ef4b10c7bcc04d0d39 | [link](https://drive.google.com/file/d/1M1NAxkDvlf0Lpxvq2FqrgNSeTMz0JL2Y/view?usp=sharing) |
|  Flickr-500k   |  11.3Mb   | e98a00e9162c3e627b8316600c0d38e5 | [link](https://drive.google.com/file/d/1H49Ua7uorj6CCVutFqrlPB9l5djaSpc0/view?usp=sharing) |
|   Flickr-1M    |  22.6Mb   | e35de83b63308b8ad4a78c4667c3e4f5 | [link](https://drive.google.com/file/d/1YK9x9KLPr_v_Doq7NqmPMZDxodG8wYEI/view?usp=sharing) |
|   Flickr-2M    |  45.3Mb   | 02b847509b2fa43d9da8e17f4ed9474c | [link](https://drive.google.com/file/d/1w8GSMXgft0d0seZgyXdUdJu_gKu9qEdW/view?usp=sharing) |
|   Flickr-5M    |  113.2Mb  | 9ab4fe89352f92c16f948e91ea2418d4 | [link](https://drive.google.com/file/d/11PyaJgdF1oTDJs6kq6gqSwI4B1xg9ccD/view?usp=sharing) |
|   Flickr-10M   |  226.4Mb  | 5486eaf81d71e65639c03a57546deba4 | [link](https://drive.google.com/file/d/1oiIvHMuOJnOsRWysLaj1eM1ePuci-FyU/view?usp=sharing) |
|   Flickr-20M   |  452.9Mb  | 9b244d66d30a51b8b5b67c2db394a03b | [link](https://drive.google.com/file/d/1Fai4qFXQhjnycSIYPQSabSLV6rNvNS-n/view?usp=sharing) |
|   Flickr-50M   |  1.11Gb   | 0c5b1b47fc80f66fa31924c8e71b49c0 | [link](https://drive.google.com/file/d/1VJdOaZQ2dTmhlt2vIlQsss3MdPIPqs8A/view?usp=sharing) |
|  Flickr-100M   |  2.21Gb   | a4ae31661bbb9feacf60a2218750f1a7 | [link](https://drive.google.com/file/d/1r0WJDmCA7KYscdR9nOXO2Ej8uTbvXlWK/view?usp=sharing) |
|  Flickr-200M   |  4.42Gb   | e82943df78b871781cc1e814405cb83c | [link](https://drive.google.com/file/d/1_1j1s1TO7sncIBRpadr-SUm4fQ9WAv1R/view?usp=sharing) |
| Flickr (+400M) |  10.03Gb  | 6a2a645c49f5983711bc3a73f0583216 | [link](https://drive.google.com/file/d/13Sa4jjOrz78xE5ZoOZY3qTuJP7k929A6/view?usp=sharing) |

### Download images
To download all images from a specific subset, run the following code.
```
cd datasets/flickrCC
python download_subset.py --root_dir /path/to/data/folder --subset ../lists/flickr-20M.tsv.gz --workers 8
```
When file lists get too large, distributed data loaders run out of memory.
To avoid duplicating the filelist within each dataloader, we convert the filelist into a memory mapped file to use during training. 
```
cd datasets/flickrCC
python convert_to_mmaps.py --subset ../lists/flickr-20M.tsv.gz --subset_size 20000000
```

# Video Datasets
Download [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) and [Krishna-CAM](https://krsingh.cs.ucdavis.edu/krishna_files/papers/krishnacam/krishnacam.html).

Download the image subsets used in the paper: [kinetics_filelist.txt](https://drive.google.com/file/d/1rYJu0dcvrqq64jV8hJsZJgvsp3B4tf1Y/view?usp=sharing) and [krishnacam_filelist.txt](https://drive.google.com/file/d/1_J3yxrp70bEqF7W7d7qgPrR0qGpJrOOr/view?usp=sharing). Place them under `datasets/lists`. 

Convert videos into frames at 10 fps using:
```
cd datasets/videoDBs
python convert_video_frames.py --video_dir /path/to/kinetics/ --frames_dir /path/to/kinetics-frames/ --workers 10
python convert_video_frames.py --video_dir /path/to/krishnacam/ --frames_dir /path/to/krishnacam-frames/ --workers 10
```

# Full ImageNet
Download [Full ImageNet](https://www.image-net.org/download.php).

Download the image subsets used in the paper: [fullimagenet_0123.memmap](https://drive.google.com/file/d/1w-rPvbknYscHDVMaldI3rtgHH7ake8Fk/view?usp=sharing), [fullimagenet_0312.memmap](https://drive.google.com/file/d/1NKUY3NxcKPvnymUN4sM3TJCqXFKSJdNK/view?usp=sharing), [fullimagenet_3120.memmap](https://drive.google.com/file/d/1rYJu0dcvrqq64jV8hJsZJgvsp3B4tf1Y/view?usp=sharing). Placed it under `datasets/lists`.
