from google.cloud import storage

bucket_name = 'gepeto-bucket'
project_name = 'neural-sol-391812'


def upload_blob(path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename('pdf/' + path)


def download_blob(path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename(path)

    return blob
