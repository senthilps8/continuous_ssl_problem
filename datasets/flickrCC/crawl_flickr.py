# Adapted from https://github.com/WordPress/openverse-catalog/blob/main/openverse_catalog/dags/providers/provider_api_scripts/flickr.py
"""
Content Provider:       Flickr

ETL Process:            Use the API to identify all CC licensed images.

Output:                 TSV file containing the images and the
                        respective meta-data.

Notes:                  https://www.flickr.com/help/terms/api
                        Rate limit: 3600 requests per hour.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
sys.path.insert(0, 'openverse-catalog/openverse_catalog/dags')

import lxml.html as html
from airflow.models import Variable
from common.licenses import get_license_info
from common.loader import provider_details as prov
from common.loader.provider_details import ImageCategory
from common.requester import DelayedRequester
from common.storage.image import ImageStore
from requests.exceptions import JSONDecodeError


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s:  %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DELAY = 1.0
LIMIT = 500
MAX_TAG_STRING_LENGTH = 2000
MAX_DESCRIPTION_LENGTH = 2000
PROVIDER = prov.FLICKR_DEFAULT_PROVIDER
API_KEY = Variable.get("API_KEY_FLICKR", default_var="0428a4d5a19a94cd943cab828deabe56")
ENDPOINT = "https://api.flickr.com/services/rest/"
PHOTO_URL_BASE = prov.FLICKR_PHOTO_URL_BASE
DATE_TYPE = "upload"
# DAY_DIVISION is an integer that gives how many equal portions we should
# divide a 24-hour period into for requesting photo data.  For example,
# DAY_DIVISION = 24 would mean dividing the day into hours, and requesting the
# photo data for each hour of the day separately.  This is necessary because
# if we request too much at once, the API will return fallacious results.
DAY_DIVISION = 288  # divide into 5 min increments
# SUB_PROVIDERS is a collection of providers within Flickr which are
# valuable to a broad audience
SUB_PROVIDERS = prov.FLICKR_SUB_PROVIDERS
NUM_BUCKETS = 10000

LICENSE_INFO = {
    "1": ("by-nc-sa", "2.0"),
    "2": ("by-nc", "2.0"),
    "3": ("by-nc-nd", "2.0"),
    "4": ("by", "2.0"),
    "5": ("by-sa", "2.0"),
    "6": ("by-nd", "2.0"),
    "9": ("cc0", "1.0"),
    "10": ("pdm", "1.0"),
}

DEFAULT_QUERY_PARAMS = {
    "method": "flickr.photos.search",
    "media": "photos",
    "safe_search": 1,  # Restrict to 'safe'
    "extras": (
        "description,license,date_upload,date_taken,owner_name,tags,o_dims,"
        "url_t,url_s,url_m,url_l,views,content_type"
    ),
    "format": "json",
    "nojsoncallback": 1,
}

delayed_requester = DelayedRequester(DELAY)


def _derive_timestamp_pair_list(date, day_division=DAY_DIVISION):
    day_seconds = 86400
    default_day_division = 48
    portion = int(day_seconds / day_division)
    # We double-check the day can be evenly divided by the requested division
    try:
        assert portion == day_seconds / day_division
    except AssertionError:
        logger.warning(
            f"day_division {day_division} does not divide the day evenly!  "
            f"Using the default of {default_day_division}"
        )
        day_division = default_day_division
        portion = int(day_seconds / day_division)

    utc_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _ts_string(d):
        return str(int(d.timestamp()))

    pair_list = [
        (
            _ts_string(utc_date + timedelta(seconds=i * portion)),
            _ts_string(utc_date + timedelta(seconds=(i + 1) * portion)),
        )
        for i in range(day_division)
    ]
    return pair_list


def _process_interval(image_store, start_timestamp, end_timestamp, date_type):
    total_pages = 1
    page_number = 1
    total_images = 0

    while page_number <= total_pages:
        logger.info(f"Processing page: {page_number} of {total_pages}")

        image_list, new_total_pages = _get_image_list(
            start_timestamp, end_timestamp, date_type, page_number
        )

        if image_list is not None:
            total_images = _process_image_list(image_store, image_list)
            logger.info(f"Total Images so far: {total_images}")
        else:
            logger.warning("No image data!  Attempting to continue")

        if new_total_pages is not None and total_pages <= new_total_pages:
            total_pages = new_total_pages

        page_number += 1

    logger.info(f"Total pages processed: {page_number}")

    return total_images


def _get_image_list(
        start_timestamp,
        end_timestamp,
        date_type,
        page_number,
        endpoint=ENDPOINT,
        max_tries=6,  # one original try, plus 5 retries
):
    image_list, total_pages = None, None
    try_number = 0
    for try_number in range(max_tries):
        query_param_dict = _build_query_param_dict(
            start_timestamp,
            end_timestamp,
            page_number,
            date_type,
        )
        response = delayed_requester.get(
            endpoint,
            params=query_param_dict,
        )

        logger.debug(f"response.status_code: {response.status_code}")
        response_json = _extract_response_json(response)
        image_list, total_pages = _extract_image_list_from_json(response_json)

        if (image_list is not None) and (total_pages is not None):
            break

    if try_number == max_tries - 1 and ((image_list is None) or (total_pages is None)):
        logger.warning("No more tries remaining. Returning Nonetypes.")

    return image_list, total_pages


def _extract_response_json(response):
    if response is not None and response.status_code == 200:
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.warning(f"Could not get image_data json.\n{e}")
            response_json = None
    else:
        response_json = None

    return response_json


def _build_query_param_dict(
        start_timestamp,
        end_timestamp,
        cur_page,
        date_type,
        api_key=API_KEY,
        license_info=None,
        limit=LIMIT,
        default_query_param=None,
):
    if license_info is None:
        license_info = LICENSE_INFO.copy()
    if default_query_param is None:
        default_query_param = DEFAULT_QUERY_PARAMS
    query_param_dict = default_query_param.copy()
    query_param_dict.update(
        {
            f"min_{date_type}_date": start_timestamp,
            f"max_{date_type}_date": end_timestamp,
            "page": cur_page,
            "api_key": api_key,
            "license": ",".join(license_info.keys()),
            "per_page": limit,
        }
    )

    return query_param_dict


def _extract_image_list_from_json(response_json):
    if response_json is None or response_json.get("stat") != "ok":
        image_page = None
    else:
        image_page = response_json.get("photos")

    if image_page is not None:
        image_list = image_page.get("photo")
        total_pages = image_page.get("pages")
    else:
        image_list, total_pages = None, None

    return image_list, total_pages


def _process_image_list(image_store, image_list):
    total_images = 0
    for image_data in image_list:
        total_images = _process_image_data(image_store, image_data)

    return total_images


def _process_image_data(image_store, image_data, sub_providers=SUB_PROVIDERS, provider=PROVIDER):
    logger.debug(f"Processing image data: {image_data}")
    image_url, height, width = _get_image_url(image_data)
    if image_url is None:
        return image_store.total_items
    license_, license_version = _get_license(image_data.get("license"))
    creator_url = _build_creator_url(image_data)
    foreign_id = image_data.get("id")
    if foreign_id is None:
        logger.warning("No foreign_id in image_data!")
    foreign_landing_url = _build_foreign_landing_url(creator_url, foreign_id)
    owner = image_data.get("owner").strip()
    source = next((s for s in sub_providers if owner in sub_providers[s]), provider)
    # filesize, filetype = _get_file_properties(image_url)
    filetype = image_url.split(".")[-1]
    return image_store.add_item(
        foreign_landing_url=foreign_landing_url,
        image_url=image_url,
        license_info=get_license_info(
            license_=license_, license_version=license_version
        ),
        foreign_identifier=foreign_id,
        width=width,
        height=height,
        # filesize=filesize,
        filetype=filetype,
        creator=image_data.get("ownername"),
        creator_url=creator_url,
        title=image_data.get("title"),
        meta_data=_create_meta_data_dict(image_data),
        raw_tags=_create_tags_list(image_data),
        source=source,
        category=_get_category(image_data),
    )


def _build_creator_url(image_data, photo_url_base=PHOTO_URL_BASE):
    owner = image_data.get("owner")
    if owner is not None:
        creator_url = _url_join(photo_url_base, owner.strip())
        logger.debug(f"creator_url: {creator_url}")
    else:
        logger.warning("No creator_url constructed!")
        creator_url = None

    return creator_url


def _build_foreign_landing_url(creator_url, foreign_id):
    if creator_url and foreign_id:
        foreign_landing_url = _url_join(creator_url, foreign_id)
        logger.debug(f"foreign_landing_url: {foreign_landing_url}")
    else:
        logger.warning("No foreign_landing_url constructed!")
        foreign_landing_url = None

    return foreign_landing_url


def _url_join(*args):
    return "/".join([s.strip("/") for s in args])


def _get_image_url(image_data):
    # prefer medium, then large, then small images
    for size in ["m", "l", "s"]:
        url_key = f"url_{size}"
        height_key = f"height_{size}"
        width_key = f"width_{size}"
        if url_key in image_data:
            return (
                image_data.get(url_key),
                image_data.get(height_key),
                image_data.get(width_key),
            )

    logger.warning("Image not detected!")
    return None, None, None


def _get_file_properties(image_url):
    """
    Get the size of the image in bytes and its filetype.
    """
    filesize, filetype = None, None
    if image_url:
        filetype = image_url.split(".")[-1]
        resp = delayed_requester.get(image_url)
        if resp:
            filesize = int(resp.headers.get("X-TTDB-L", 0))
    return (
        filesize if filesize != 0 else None,
        filetype if filetype != "" else None,
    )


def _get_license(license_id, license_info=None):
    if license_info is None:
        license_info = LICENSE_INFO.copy()
    license_id = str(license_id)

    if license_id not in license_info:
        logger.warning("Unknown license ID!")

    license_, license_version = license_info.get(license_id, (None, None))

    return license_, license_version


def _create_meta_data_dict(image_data, max_description_length=MAX_DESCRIPTION_LENGTH):
    meta_data = {
        "pub_date": image_data.get("dateupload"),
        "date_taken": image_data.get("datetaken"),
        "views": image_data.get("views"),
    }
    description = image_data.get("description", {}).get("_content", "")
    logger.debug(f"description: {description}")
    if description.strip():
        try:
            description_text = " ".join(
                html.fromstring(description).xpath("//text()")
            ).strip()[:max_description_length]
            meta_data["description"] = description_text
        except (TypeError, ValueError, IndexError) as e:
            logger.warning(f"Could not parse description {description}!\n{e}")

    return {k: v for k, v in meta_data.items() if v is not None}


def _create_tags_list(image_data, max_tag_string_length=MAX_TAG_STRING_LENGTH):
    raw_tags = None
    # We limit the input tag string length, not the number of tags,
    # since tags could otherwise be arbitrarily long, resulting in
    # arbitrarily large data in the DB.
    raw_tag_string = image_data.get("tags", "").strip()[:max_tag_string_length]
    if raw_tag_string:
        # We sort for further consistency between runs, saving on
        # inserts into the DB later.
        raw_tags = sorted(list(set(raw_tag_string.split())))

    return raw_tags


def _get_category(image_data):
    """
    Flickr has three types:
        0 for photos
        1 for screenshots
        3 for other
    Treating everything different from photos as unknown.
    """
    if "content_type" in image_data and image_data["content_type"] == "0":
        return ImageCategory.PHOTOGRAPH.value
    return None


def compress(date_str):
    output_dir = f'meta_by_date/{date_str}/'
    all_tsv = [f"{root}/{fn}" for root, subdirs, files in os.walk(output_dir) for fn in files if fn.endswith('tsv')]
    if len(all_tsv) == 0:
        return

    import gzip
    with gzip.open(f'{output_dir}/flickr_images_{date_str}.tsv.gz', 'wb') as fp:
        deduplicated, fids = [], set()
        for tsv_file in all_tsv:
            contents = list(open(tsv_file, 'rb'))
            for data in contents:
                fid = data.decode('utf-8').split('\t')[0]

                # Assign random bucket for local storage
                import random
                bidx = random.randint(0, NUM_BUCKETS-1)
                data_list = data.decode('utf-8').strip().split('\t')
                data_list.append(f"{bidx:04d}")
                data = ('\t'.join(data_list)+'\n').encode('utf-8')

                if fid not in fids:
                    deduplicated.append(data)
                    fids.add(fid)

        fp.write(b"".join(deduplicated))

    # Remove original uncompressed tsv files
    for tsv_file in all_tsv:
        os.remove(tsv_file)


def crawl_date(date_str, crawl_folder):
    logger.info(f"Processing Flickr API for date: {date_str}")

    output_dir = f'{crawl_folder}/{date_str}/'
    if os.path.isfile(f"{output_dir}/done"):
        logger.info("Date already processed!")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_store = ImageStore(provider=PROVIDER, output_dir=output_dir, buffer_length=1000)

    timestamp_pairs = _derive_timestamp_pair_list(date_str)
    date_type = DATE_TYPE

    for start_timestamp, end_timestamp in timestamp_pairs:
        _process_interval(image_store, start_timestamp, end_timestamp, date_type)
    total_images = image_store.commit()

    compress(date_str)

    open(f"{output_dir}/done", 'w').close()
    logger.info(f"Total images: {total_images}")
    logger.info("Terminated!")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_date', default='2013-01-01', help='Begin date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default='2013-02-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--crawl_folder', default='meta_by_date', help='Folder to store crawl data.')
    parser.add_argument('--workers', default=0, type=int, help='Number of parallel workers')
    args = parser.parse_args()

    start_date = datetime.strptime(args.init_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    all_dates = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]
    from functools import partial
    crawl_fcn = partial(crawl_date, crawl_folder=args.crawl_folder)

    if args.workers == 0:
        for d in all_dates:
            crawl_fcn(d)
    else:
        pool = mp.Pool(args.workers)
        pool.map(crawl_fcn, all_dates)


if __name__ == '__main__':
    main()
