from google.cloud import storage

bucket_name = 'gepeto1'
project_name = 'brave-drive-388410'


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
