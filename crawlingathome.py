import itertools
import os
import pickle
import shutil
import time
from contextlib import asynccontextmanager
from copy import copy
from glob import glob
from typing import Callable, List
from urllib.parse import urljoin, urlparse
from uuid import uuid1

import asks
import pandas as pd
import regex
import tractor
import trio
import ujson
from async_generator import aclosing
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486
session = asks.Session(connections=64)


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def remove_bad_chars(text):
    return regex.sub(r"\p{Cc}|\p{Cs}", "", text)


# Thank you tractor maintainer
@asynccontextmanager
async def worker_pool(workers=4, data=None):
    """Though it's a trivial special case for ``tractor``, the well
    known "worker pool" seems to be the defacto "but, I want this
    process pattern!" for most parallelism pilgrims.
    Yes, the workers stay alive (and ready for work) until you close
    the context.
    """
    async with tractor.open_nursery() as tn:

        portals = []
        snd_chan, recv_chan = trio.open_memory_channel(len(data))

        for i in range(workers):

            # this starts a new sub-actor (process + trio runtime) and
            # stores it's "portal" for later use to "submit jobs" (ugh).
            portals.append(
                await tn.start_actor(
                    f"worker_{i}",
                    enable_modules=[__name__],
                )
            )

        async def _map(
            worker_func: Callable[[int], bool], sequence: List[int]
        ) -> List[bool]:

            # define an async (local) task to collect results from workers
            async def send_result(func, value, portal):
                await snd_chan.send((value, await portal.run(func, data=value)))

            async with trio.open_nursery() as n:

                for value, portal in zip(sequence, itertools.cycle(portals)):
                    n.start_soon(send_result, worker_func, value, portal)

                # deliver results as they arrive
                for _ in range(len(sequence)):
                    yield await recv_chan.receive()

        # deliver the parallel "worker mapper" to user code
        yield _map

        # tear down all "workers" on pool close
        await tn.cancel()


def parse_wat(content, start, line_count, first_sample_id):
    import ftfy
    import pycld2 as cld2

    valid_data = []
    content.seek(start)
    sample_id = first_sample_id
    for _ in range(line_count):
        line = content.readline()
        if "IMG@" not in line:
            continue
        line_str = line.strip()
        data = ujson.loads(line_str)
        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
            "HTML-Metadata"
        ]["Links"]
        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )  # get base url
        license = "?"
        for e in linklist:
            if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                license = e["url"]
            if "alt" not in e:
                continue
            url = e["url"]
            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            if url.endswith(".svg") or url.endswith(".gif") or "data:image" in url:
                continue
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                sample_id += 1
                valid_data.append((url, alt_text, license, sample_id))
    return [
        t for t in {tuple(i) for i in valid_data}
    ]  # Remove duplicate tuple from list


def process_img_content(response, alt_text, license, sample_id):
    img_output_folder = "save/images/"
    if "content-type" in response.headers:
        if "image/" not in response.headers["content-type"]:
            return
        filetype = (
            response.headers["content-type"].split("/")[-1].split(";")[0]
        ).strip()  # Unreliable, maybe get filetype from content?
    else:
        url_path = urlparse(response.url).path
        filetype = os.path.splitext(url_path)[1].strip()

    if filetype not in ["jpeg", "jpg", "png"] or len(response.content) < 5000:
        return

    out_fname = img_output_folder + str(sample_id) + "." + filetype.strip(".")
    try:
        img_data = response.content  # Raise KeyError
        with open(out_fname, "wb") as f:
            f.write(img_data)
        pil_image = Image.open(out_fname)  # Raise UnidentifiedImageError
    except (KeyError, UnidentifiedImageError) as e:
        if os.path.exists(out_fname):
            os.remove(out_fname)
        return
    width, height = pil_image.size
    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(data):
    url, alt_text, license, sample_id = data
    try:
        return process_img_content(
            await session.get(url, timeout=5), alt_text, license, sample_id
        )
    except:
        return None

async def dl_wat(parsed_data):
    processed_samples = []
    async with worker_pool(data=parsed_data) as actor_map:
        async with aclosing(actor_map(request_image, parsed_data)) as results:
            async for _, result in results:
                if result != None:
                    processed_samples.append(result)
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]
    import clip_filter

    clip = clip_filter.CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    nsfw_filters = clip.filter(img_embedding, clip.categories)
    underage_filters = clip.filter(img_embedding, clip.underaged_categories)
    animal_filters = clip.filter(img_embedding, clip.animal_categories)
    tmp_embed = copy(img_embedding)
    for i, (nsfw_prob, underage_prob, animal_prob, img_embed) in enumerate(
        zip(nsfw_filters, underage_filters, animal_filters, tmp_embed)
    ):
        df.at[i, "similarity"] = similarities[i]
        df.at[i, "NSFW"] = "UNSURE"

        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        # If image is nsfw and text is containing underaged or image is containing underage or image is containing animal
        is_nsfw_underaged = (
            df.at[i, "NSFW"] == "NSFW" or df.at[i, "NSFW"] == "UNSURE"
        ) and (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
            or animal_prob[0] > 20
        )

        # Remove image containing underage and not similar image-alttext
        if similarities[i] < sim_threshold or is_nsfw_underaged:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
    df.reset_index(drop=True, inplace=True)
    return df, img_embedding


def df_tfrecords(df, output_fname):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


def upload_gdrive(output_filename):
    import requests

    client_id = (
        "648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com"
    )
    client_secret = "HZ4Zw-_jVJ-3mwicz1NM5W5x"
    refresh_token = "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA"

    def refresh_gdrive_token():
        params = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }

        authorization_url = "https://www.googleapis.com/oauth2/v4/token"

        r = requests.post(authorization_url, data=params)

        if r.ok:
            return r.json()["access_token"]
        else:
            return None

    access_t = refresh_gdrive_token()
    headers = {"Authorization": "Bearer " + access_t}
    para = {
        "name": output_filename.split("/")[-1],
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"],
    }

    files = {
        "data": ("metadata", ujson.dumps(para), "application/json; charset=UTF-8"),
        "file": ("application/zip", open(output_filename, "rb")),
    }
    requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )


class FileData:
    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, "r") as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1

    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length


if __name__ == "__main__":
    import crawlingathome_client as cah

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = "Wikidepia"
    CRAWLINGATHOME_SERVER_URL = "http://178.63.68.247:8181/"

    client = cah.init(
        url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
    )
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    def cah_log(msg):
        for _ in range(3):
            try:
                return client.log(msg)
            except crawlingathome_client.errors.ServerError:
                time.sleep(1)
                continue

    while client.jobCount() > 0:
        start = time.time()
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        if os.path.exists(".tmp"):
            shutil.rmtree(".tmp")

        os.mkdir(output_folder)
        os.mkdir(img_output_folder)
        os.mkdir(".tmp")

        client.newJob()
        client.downloadShard()
        first_sample_id = int(client.start_id)
        last_sample_id = int(client.end_id)
        shard_of_chunk = client.shard_piece  # TODO

        fd = FileData("shard.wat")
        if shard_of_chunk == 0:
            start_index = fd[0]
        if shard_of_chunk == 1:
            start_index = fd[int(len(fd) * 0.5)]
        lines = int(len(fd) * 0.5)

        out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
        cah_log("Processing shard")
        with open("shard.wat", "r") as infile:
            parsed_data = parse_wat(infile, start_index, lines, first_sample_id)

        cah_log("Downloading images")
        dlparse_df = trio.run(dl_wat, parsed_data)
        dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")

        cah_log("Dropping NSFW keywords")
        filtered_df, img_embeddings = df_clipfilter(dlparse_df)
        filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
        img_embeds_sampleid = {}
        for i, img_embed_it in enumerate(img_embeddings):
            dfid_index = filtered_df.at[i, "SAMPLE_ID"]
            img_embeds_sampleid[str(dfid_index)] = img_embed_it
        with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
            pickle.dump(img_embeds_sampleid, f)

        cah_log("Saving TFRs")
        df_tfrecords(
            filtered_df,
            f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
        )
        # upload_gdrive(output_folder + "image_embeddings.pkl")
        # upload_gdrive(output_folder + "images.tfrecord")
        client._markjobasdone(len(filtered_df))
        print(f"[crawling@home] jobs completed in {round(time.time() - start)} seconds")
