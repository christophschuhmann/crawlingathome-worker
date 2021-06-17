import gc
import os
import pickle
import shutil
import time
from glob import glob
from urllib.parse import urljoin, urlparse
from uuid import uuid1

import regex
import trio
import ujson
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


def uploadGdrive(output_filename):
    #output_filename = Path(output_filename).name
 
    access_t = refreshToken("648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com","HZ4Zw-_jVJ-3mwicz1NM5W5x", "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA")                                                                                                   
 
    headers = {"Authorization": "Bearer " + access_t} #put ur access token after the word 'Bearer '
 
    para = {
        "name": output_filename.split("/")[-1], # file name to be uploaded
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"] # make a folder on drive in which you want to upload files; then open that folder; the last thing in present url will be folder id
    }
    
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': ('application/zip',open( output_filename , "rb")) # replace                        
    }
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files
    )
 


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def remove_bad_chars(text):
    return regex.sub(r"\p{Cc}|\p{Cs}", "", text)


def imgfiles_to_embeddings(list_of_files, batch_size, model, preprocess, device):
      if batch_size<2:
        print("Minimal batch_size is 2 ")
        return []

      import numpy as np
      from PIL import Image

      import time
      import torch.nn as nn

      import torch
      #import clip
      import os

      counter_samples =0

      list_of_arrays_to_concat = []
      list_of_tokenized_text_arrays =[]
      list_of_arrays_to_concat = []
      list_of_image_arrays =[]
      list_of_tokenized_text_arrays =[]
      img_embeddings= []


      for img_path in list_of_files:

        try:
          new_image_array = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except:
          new_image_array = preprocess(Image.new("RGB", (300, 300), (255, 255, 255))).unsqueeze(0).to(device)


        if counter_samples%batch_size ==0:
          image_array =new_image_array
          #tokenized_text_np_array = tokenized_text_np_array_new_sample
          counter_samples +=1
          continue
        else:
          image_array =  torch.cat((image_array,new_image_array), 0)
          counter_samples +=1
          #print(image_array.shape)



        if counter_samples%batch_size ==0:
            with torch.no_grad():
              image_features = model.encode_image(image_array)


              for i in range (image_features.shape[0]):
                  img_embeddings.append(torch.reshape(image_features[i], (1, 512)))
                  #img_embeddings.append(image_features[i])
                  #print(torch.reshape(image_features[i], (1, 512)) .shape)
      with torch.no_grad():
          image_features = model.encode_image(image_array)
          for i in range (image_features.shape[0]):
            img_embeddings.append(torch.reshape(image_features[i], (1, 512)))
            #img_embeddings.append(image_features[i])

      #print(len(img_embeddings))
      #print(img_embeddings[0].shape)
      return img_embeddings



def parse_wat(content, start, line_count):
    import ftfy
    import pycld2 as cld2

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
                valid_data.append((url, alt_text, license))
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


async def request_image(datas, start_sampleid):
    import asks

    tmp_data = []
    session = asks.Session(connections=64)

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


async def dl_wat(valid_data, first_sample_id):
    import pandas as pd
    import tractor

    # Download every image available
    processed_samples = []
    async with tractor.open_nursery() as n:
        for i, data in enumerate(chunk_using_generators(valid_data, 65356)):
            await n.run_in_actor(
                request_image, datas=data, start_sampleid=i * 65356 + first_sample_id
            )

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
        img_sample_id = img_ids_by_filepath[path]
        image_embedding_dict[img_sample_id] = img_emb_list[c]

        c +=1


    untokenized_texts=[]

    tokenized_texts=[]
    sample_ids_tokenized_texts=[]

    text_embedding_list = []
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
                df_image["SAMPLE_ID"].encode("utf_8"),
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

import sys, os


def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class FileData:
    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, 'r') as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1

    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length

if __name__ == "__main__":
    import time
    import crawlingathome_client as cah

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = "Kris"
    CRAWLINGATHOME_SERVER_URL = "http://crawlingathome.duckdns.org/"
    import logging
    logging.basicConfig(filename="log.log", level=logging.INFO)
    client = cah.init(
        url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
    )
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"
    os.system("ulimit -n 120000")
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
            logging.info('dirs cleared')
            client.newJob()
            client.downloadShard()
            first_sample_id = int(client.start_id)
            last_sample_id = int(client.end_id)
            shard_of_chunk = client.shard_piece  # TODO


            fd = FileData('shard.wat')
            
            if shard_of_chunk == 0:
                start_index = fd[ int(len(fd)*0.995) ]#fd[0]
            if shard_of_chunk == 1:
                start_index = fd[ int(len(fd)*0.995) ] #fd[ int(len(fd)*0.5) ]
            lines = int(len(fd)*0.5)
            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            client.log("Processing shard")
            with open("shard.wat", "r") as infile:
                parsed_data = parse_wat(infile, start_index, lines)

            client.log("Downloading images")
            dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id)
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")

            print(f"Downloads completed in {round(time.time() - start)} seconds")
            logging.info("DL completed {a}".format(a=  round(time.time() - start)))
            print("Filtering begins")
            logging.info("Samples before CLIP {a}".format(a=   len(dlparse_df) ))
            filtered_df, img_embeddings = df_clipfilter(dlparse_df)
            print(len(dlparse_df))
            print(len(filtered_df))

            
            logging.info("Samples after CLIP {a}".format(a=   len(filtered_df) ))

            filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
                pickle.dump(img_embeddings, f)

            client.log("Saving TFRs")
            print("before df_tfrecords")
            df_tfrecords(
                filtered_df,
                "crawling_at_home_"+ 'FIRST_SAMPLE_ID_IN_SHARD_'+str(first_sample_id)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(last_sample_id)+"_"+str(shard_of_chunk)+".tfrecord"
            )
            print("after df_tfrecords")
            from pathlib import Path
            saves = Path("./save")
 
            client.log("Uploading CSV")
            uploadGdrive(f"./save/FIRST_SAMPLE_ID_IN_SHARD_{first_sample_id}_LAST_SAMPLE_ID_IN_SHARD_{last_sample_id}_"+str(shard_of_chunk)+".csv")
 
            client.log("Uploading TFRECORD")
            tfrecords = [*saves.glob("*.tfrecord")]
            for f in tfrecords:
               uploadGdrive(str(f))
        
            client.log("Uploading Image Embeddings")
            uploadGdrive(f"./save/image_embedding_dict-FIRST_SAMPLE_ID_IN_SHARD_{first_sample_id}_LAST_SAMPLE_ID_IN_SHARD_{last_sample_id}_"+str(shard_of_chunk)+".pkl")
 
           
            #client._markjobasdone(len(filtered_df))

            print(f"[crawling@home] jobs completed in {round(time.time() - start)} seconds")

            logging.info("Job completed {a}".format(a=  round(time.time() - start)))
        
        except Exception as e: # work on python 3.x
            logging.error('Error: '+ str(e))
            print('Error: '+ str(e))
            
            try:
               client.bye()
            except:
                pass

            time.sleep(3)
            import crawlingathome_client as cah

            client = cah.init(
                url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
            )
            output_folder = "./save/"
            csv_output_folder = output_folder
            img_output_folder = output_folder + "images/"
            os.system("ulimit -n 120000")
            
            continue
   
