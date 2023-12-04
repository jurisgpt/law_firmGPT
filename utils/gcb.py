from google.cloud import storage

# Set up the storage client using the JSON key file
client = storage.Client.from_service_account_json('/root/workspace/gkey/juris-gpt-key.json')

# Define the name of the bucket and the name of the file in Google Cloud Storage
bucket_name = 'juris-tsunzu-gpt-bucket'
blob_name = 'test/'

# Convert your data into bytes
data = b'your data in bytes'

# Create a new blob in the bucket and upload the data
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.upload_from_string('')

print(f'Data uploaded to {bucket_name}/{blob_name}')
