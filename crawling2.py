import argparse
import gc
import os
import pickle
import random
import shutil
import time
from copy import copy
from glob import glob
from io import BytesIO
from urllib.parse import urljoin
from uuid import uuid1

import tractor
import trio
import ujson
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count):
    import ftfy
    import pycld2 as cld2

    blocklist = open("blocklist-domain.txt").read().splitlines()
    valid_data = []
    content.seek(start)
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
            if any(
                x in url for x in [".svg", ".gif", "data:image", "javascript:"]
            ) or any(bl in url for bl in blocklist):
                continue
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                valid_data.append((url, alt_text, license))
    return [
        t for t in {tuple(i) for i in valid_data}
    ]  # Remove duplicate tuple from list


def process_img_content(response, alt_text, license, sample_id):
    img_output_folder = "save/images/"

    try:
        if len(response.content) < 5000:
            return
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            im_format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{im_format.lower()}"
            if im_format not in ["JPEG", "JPG", "PNG"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid):
    import asks

    tmp_data = []
    session = asks.Session(connections=165)
    session.headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        try:
            proces = process_img_content(
                await session.get(url, timeout=5), alt_text, license, sample_id
            )
            if proces is not None:
                tmp_data.append(proces)
        except Exception:
            return

    async with trio.open_nursery() as n:
        for data in datas:
            n.start_soon(_request, data, start_sampleid)
            start_sampleid += 1

    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return


def dl_wat(valid_data, first_sample_id):
    import multiprocessing as mp

    import pandas as pd

    # Download every image available
    processed_samples = []
    n_processes = mp.cpu_count()

    if n_processes == 1:
        trio.run(request_image, valid_data, first_sample_id)
    else:
        async def _runtractor():
            async with tractor.open_nursery() as n:
                chunk_size = len(valid_data)//n_processes + 1
                for i, data in enumerate(chunk_using_generators(valid_data, chunk_size)):
                    await n.run_in_actor(
                        request_image, datas=data, start_sampleid = first_sample_id + i*chunk_size
                    )
        trio.run(_runtractor)

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def df_clipfilter(df):

    import torch.nn as nn

    import torch
    import clip
    from PIL import Image
    import glob
    from pathlib import Path
    similarity_threshold = 0.3

    img_output_folder = "save/images/"

    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    print ("len(df) before filtering with clip"+str(len(df)))

    img_files = glob.glob(img_output_folder + "*.*")
    img_files_ids ={}
    img_ids_by_filepath={}
    for img_path in img_files:
        path = Path(img_path)
        path.name
        img_files_ids[path.stem]= img_path
        img_ids_by_filepath[img_path] = path.stem



    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    batch_size = 128 # for GPU 512 or 1024
    img_emb_list= imgfiles_to_embeddings(img_files, batch_size, model, preprocess, device )

    image_embedding_dict = {}

    c= 0
    for path in img_files:
        if c%5000 == 0:
            client.log("Encoding images with CLIP")
            print("Embbedded images: "+ str(c))
        img_sample_id = img_ids_by_filepath[path]
        image_embedding_dict[img_sample_id] = img_emb_list[c]

        c +=1


    untokenized_texts=[]

    tokenized_texts=[]
    sample_ids_tokenized_texts=[]

    text_embedding_list = []
    client.log("Encoding Texts with CLIP")
    for row_index, row in df.iterrows():
        untokenized_texts.append (str( df.at[row_index,'TEXT']) [:75])
        sample_ids_tokenized_texts.append (df.at[row_index,'SAMPLE_ID'])
        if row_index% batch_size ==0 and row_index >0:
            
            tokenized_texts = clip.tokenize(untokenized_texts).to(device)
            with torch.no_grad():
              text_embeddings = model.encode_text(tokenized_texts)
            for i in range(text_embeddings.shape[0]):
              text_embedding_list.append(text_embeddings[i])

            untokenized_texts=[]

    if len(untokenized_texts)>0:
        tokenized_texts = clip.tokenize(untokenized_texts).to(device)

        with torch.no_grad():
          text_embeddings = model.encode_text(tokenized_texts)
        for i in range(text_embeddings.shape[0]):
          text_embedding_list.append(text_embeddings[i])
        untokenized_texts=[]

    #### NSFW detector categories text embeddings

    #0-18 /first 19 are not NSFW
    nsfw_text_categories = ["neutral","selfie", "illustration, drawng", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]

    nsfw_text_tokenized = clip.tokenize(nsfw_text_categories).to(device)
    nsfw_text_features =[]
    with torch.no_grad():
      nsfw_text_embed = model.encode_text(nsfw_text_tokenized)

    for i in range(nsfw_text_embed.shape[0]):
        nsfw_text_features.append(nsfw_text_embed[i])

    listofzeros = ["-"] * len(df)

    df["NSFW"]=listofzeros



    #first 4 are underaged, 0-3
    underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]


    underaged_text_tokenized = clip.tokenize(underaged_categories).to(device)
    underaged_text_features =[]
    with torch.no_grad():
      underaged_text_embed = model.encode_text(underaged_text_tokenized)

    for i in range(underaged_text_embed.shape[0]):
        underaged_text_features.append(underaged_text_embed[i])


    #0-20 /first 21 are not animals
    animal_categories = ["lifelss object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]

    animal_text_tokenized = clip.tokenize(animal_categories).to(device)
    animal_text_features =[]
    with torch.no_grad():
      animal_text_embed = model.encode_text(animal_text_tokenized)

    for i in range(animal_text_embed.shape[0]):
        animal_text_features.append(animal_text_embed[i])


    # given an iterable of pairs return the key corresponding to the greatest value
    def argmax(pairs):
        return max(pairs, key=lambda x: x[1])[0]

    # given an iterable of values return the index of the greatest value
    def argmax_index(values):
        return argmax(enumerate(values))


    listofzeros = [0.0] * len(df)

    df["similarity"]=listofzeros

    #image_embedding_dict= {}
    #print ("len(df)"+str(len(df)))

    img_dict_counter= 0
    #print ("len(df) before 1st for row_index, row in df.iterrows():"+str(len(df)))


    #client.log("Dropping NSFW Keywords")


    for row_index2, row2 in df.iterrows():
        if str(df.at[row_index2,'TEXT']).lower().find("sex") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("nude") !=-1  or  str(df.at[row_index2,'TEXT']).lower().find("sexy") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("fuck") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("orgasm") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("porn") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lesbian") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lust") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("pussy") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("bdsm") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("titts") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("horny") !=-1   or str(df.at[row_index2,'TEXT']).lower().find("nacked") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("boops") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("erotic") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lingerie") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("penis") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("dick") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("cock") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("dig") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("clit") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("nipple") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("gay") !=-1  :

            if str(df.at[row_index2,'TEXT']).lower().find("teen") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("kid") !=-1  or  str(df.at[row_index2,'TEXT']).lower().find("child") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("baby") !=-1 :

                #print(###########NSFW KEYWORD DROP##############)

                #print (df.at[row_index2,'TRANSLATION']))
                df = df.drop(row_index2)
                continue

    similarity_counter= 0
    for row_index, row in df.iterrows():
        try:


            if row_index % 100 ==0:
                pass
                #print("row_index: "+ str(row_index))
                #client.log(f"Removing NFSW: {row_index} / ?")

            sample_id = df.at[row_index,'SAMPLE_ID']
            index_of_row_in_list= sample_ids_tokenized_texts.index(sample_id)

            if index_of_row_in_list==-1:
                df = df.drop(row_index)
                continue

            current_text_embedding = text_embedding_list[index_of_row_in_list]
            current_image_embedding = image_embedding_dict[str(sample_id)]

            similarity= float (cosine_similarity(torch.reshape(current_text_embedding, (1, 512)) , current_image_embedding ))
            #print(df.at[row_index,'TEXT'])
            #print(df.at[row_index,'URL'])
            #print("similarity:")

            #print(similarity)
            if similarity > similarity_threshold:
                df.at[row_index,'similarity'] = similarity
                similarity_counter +=1



                #0-18 /first 19 are not NSFW
                nsfw_text_categories = ["neutral","selfie", "illustration, drawng", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]
                #nsfw_text_features = model.encode_text(nsfw_text_categories)
                similarities=[]

                for i in range(len(nsfw_text_features)):
                    similarity= float (cosine_similarity(torch.reshape(nsfw_text_features[i], (1, 512)) , current_image_embedding ))
                    similarities.append( similarity )

                #print(similarities)

                argmax1= argmax_index(similarities)
                most_likely= nsfw_text_categories[argmax1]
                #print ("most_likely")
                #print (most_likely)


                nsfw_text_categories.pop(argmax_index(similarities))
                similarities.pop(argmax_index(similarities))
                argmax2= argmax_index(similarities)
                second_likely = nsfw_text_categories[argmax_index(similarities)]

                if argmax1 <19 and argmax2<19:
                    df.at[row_index,'NSFW'] = "UNLIKELY"
                elif argmax1 <19 and argmax2>=19:
                    df.at[row_index,'NSFW'] = "UNSURE"
                elif argmax2 <19 and argmax1>=19:
                    df.at[row_index,'NSFW'] = "UNSURE"
                elif argmax1 >=19 and argmax2>=19:
                    df.at[row_index,'NSFW'] = "NSFW"



                ####underaged check
                if df.at[row_index,'NSFW'] != "UNLIKELY":

                    #keyword check
                    if str(df.at[row_index,'TEXT']).lower().find("teen") !=-1 or str(df.at[row_index,'TEXT']).lower().find("kid") !=-1  or  str(df.at[row_index,'TEXT']).lower().find("child") !=-1 or str(df.at[row_index,'TEXT']).lower().find("baby") !=-1 :
                        df = df.drop(row_index)
                        #print(###########NSFW KEYWORD DROP##############)
                        #print (df.at[row_index,'TEXT']))
                        continue

                    #first 4 are underaged, 0-3
                    underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age", "drawing, logo, clip art", "illustration, cartoon", "captcha, screen", "food, eating, meal, drink", "car"]

                    similarities=[]

                    for i in range(len(underaged_text_features)):
                        #similarities.append( cosine_similarity([underaged_text_features[i][0]], [current_image_embedding[0][0]]) )

                        similarity= float (cosine_similarity(torch.reshape(underaged_text_features[i], (1, 512)) , current_image_embedding ))
                        similarities.append( similarity )

                    argmax1= argmax_index(similarities)
                    #print("argmax1")
                    #print(argmax1)
                    most_likely= underaged_categories[argmax1]

                    #print ("most_likely")

                    #print (most_likely)

                    underaged_categories.pop(argmax_index(similarities))
                    similarities.pop(argmax_index(similarities))
                    argmax2= argmax_index(similarities)
                    #print("argmax2")
                    #print(argmax2)
                    second_likely = underaged_categories[argmax_index(similarities)]
                    #print(second_likely)
                    if argmax1 <4 or argmax2 <4:
                        #print( df.at[row_index,'URL'] )
                        del image_embedding_dict[str(sample_id)]
                        df = df.drop(row_index)

                        #print("dropped cause NSFW and eventually underaged")

                        continue


                ####animal check
                if df.at[row_index,'NSFW'] != "UNLIKELY":

                    #0-20 /first 21 are not animals
                    animal_categories = ["lifelss object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]

                    similarities=[]


                    for i in range(len(animal_text_features)):
                        #similarities.append( cosine_similarity([animal_text_features[i][0]], [current_image_embedding[0][0]]) )
                        similarity= float (cosine_similarity(torch.reshape(animal_text_features[i], (1, 512)) , current_image_embedding ))
                        similarities.append( similarity )
                    #print ("most_likely")

                    #print (most_likely)

                    argmax1= argmax_index(similarities)
                    most_likely= animal_categories[argmax1]


                    #print(second_likely)
                    if argmax1 >20:

                        del image_embedding_dict[str(sample_id)]

                        df = df.drop(row_index)
                        #print("dropped cause NSFW and eventually animal")

                        continue

            else:
                del image_embedding_dict[str(sample_id)]
                df = df.drop(row_index)
                continue

        except Exception as e:
            #print("dropped sample: "+str(df.at[row_index,'SAMPLE_ID']))
            print(e)
            #print( "embedding error")

            try:
                df = df.drop(row_index)
            except:
                pass
                #print("WEIRD ERROR")
            continue


    df.reset_index(drop=True, inplace=True)
    return df, image_embedding_dict




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
    parser = argparse.ArgumentParser(
        description = 'Crawling@Home Worker Script'
    )

    parser.add_argument('--name', '-n', type=str, default="ARKseal", help='Your name')
    parser.add_argument('--url', '-u', type=str, default="https://api.gagepiracy.com:4483/", help='The Crawling Server')

    args = parser.parse_args()

    import crawlingathome_client as cah

    client = cah.init(
        url=args.url, nickname=args.name
    )

    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    while client.jobCount() > 0:
        try:
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
            shard_of_chunk = client.shard_piece

            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            print(
                f"[crawling@home] shard identification {out_fname}"
            )  # in case test fails, we need to remove bad data
            client.log("Processing shard")

            fd = FileData("shard.wat")

            if shard_of_chunk == 0:
                start_index = fd[0]
            if shard_of_chunk == 1:
                start_index = fd[int(len(fd) * 0.5)]

            lines = int(len(fd) * 0.5)

            with open("shard.wat", "r") as infile:
                parsed_data = parse_wat(infile, start_index, lines)

            random.shuffle(parsed_data)

            client.log("Downloading images")
            dlparse_df = dl_wat(parsed_data, first_sample_id)
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            print (f"[crawling@home] Downloaded {len(dlparse_df)} in {round(time.time() - start)} seconds")
            print (f"[crawling@home] Download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")

            client.log("Dropping NSFW keywords")
            start2 = time.time()
            filtered_df, img_embeddings = df_clipfilter(dlparse_df)
            filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            print (f"[crawling@home] CLIP filtered {len(filtered_df)} in {round(time.time() - start2)} seconds")
            print (f"[crawling@home] CLIP efficiency {len(dlparse_df)/(time.time() - start2)} img/sec")

            img_embeds_sampleid = {}
            for i, img_embed_it in enumerate(img_embeddings):
                dfid_index = filtered_df.at[i, "SAMPLE_ID"]
                img_embeds_sampleid[str(dfid_index)] = img_embed_it
            with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
                pickle.dump(img_embeds_sampleid, f)

            client.log("Saving TFRs")
            df_tfrecords(
                filtered_df,
                f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
            )
            upload_gdrive(f"{output_folder}image_embedding_dict-{out_fname}.pkl")
            upload_gdrive(
                f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord"
            )
            upload_gdrive(output_folder + out_fname + ".csv")
            client._markjobasdone(len(filtered_df))
            print(f"[crawling@home] job completed in {round(time.time() - start)} seconds")
            print(f"[crawling@home] job efficiency {len(filtered_df)/(time.time() - start)} pairs/sec")
        except KeyboardInterrupt:
            print("[crawling@home] stopping crawler")
            break
        except Exception as ex:
            print(f"[crawling@home] ERROR: {ex}")
    client.bye()
