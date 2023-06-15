from google.cloud import storage

bucket_name = 'gepeto-bucket'


def upload_blob(path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename(path)


def download_blob(path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename(path)


upload_blob("pdf/last_pdf.txt")
