import os
import sys
import requests
import csv


DATA_DIR = '/tmp/stackline/'  # where the images will be stored


def download_image(data_dir, key, url, category=None):
    """
    download images to disk from url

    :param key: key for the image file, used as the file name
    :type key: str
    :param url: url to the image file
    :type url: str
    :param category: categeory for the image, used to save to a class directory
    :type category: str
    :return: None
    :rtype: None
    """
    print("\tASIN:{}, URL:{}".format(key, url))

    file_type = url.split('.')[-1]
    filename = "{}/{}.{}".format(category, key, file_type) if category else "{}.{}".format(key, file_type)
    image = requests.get(url)
    if image.status_code == 200:
        with open(data_dir + filename, 'wb') as file:
            file.write(image.content)
    else:
        print("ERROR {}: {}".format(image.status_code, url))


def main(dataset, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(data_dir + dataset):
        os.mkdir(data_dir + dataset)

    data_dir += dataset + "/"

    with open('data/processed/{}.csv'.format(dataset)) as file:
        reader = csv.reader(file)
        next(reader)  # skip the headers
        for row in reader:
            if dataset == 'training':
                asin = row[1]
                url = row[-2]
                category = row[2]
                if not os.path.exists(data_dir + category + "/"):
                    os.mkdir(data_dir + category + "/")
                download_image(data_dir, asin, url, category)

            elif dataset == 'validation':
                asin = row[1]
                url = row[-1]
                download_image(data_dir, asin, url)


if __name__ == "__main__":
    dataset = "training"
    try:
        dataset = sys.argv[1]
    except:
        pass
    main(dataset, DATA_DIR)
