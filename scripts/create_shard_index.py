import boto3
import json
import sys

def create_shard_index(bucket_name, prefix):
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Define the output filename based on the prefix
    output_file = f"{prefix}-shards.json"
    
    # Prepare the final JSON structure
    shard_index = {
        "__kind__": "wids-shard-index-v1",
        "name": prefix,
        "wids_version": 1,
        "shardlist": []
    }
    
    # List files in the specific S3 prefix, appending '/' to list contents properly if needed
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{prefix}/" if not prefix.endswith('/') else prefix)

    # Sort the contents by the 'Key' to ensure processing order
    contents = sorted(response.get('Contents', []), key=lambda x: x['Key'])
    
    # Process each file found in sorted order
    for item in contents:
        key = item['Key']
        if key.endswith('_stats.json'):
            # Get the content of the file
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            stats_data = json.loads(obj['Body'].read().decode('utf-8'))
            
            # Create the shard entry using the 'successes' field
            shard_entry = {
                "url": f"s3://{bucket_name}/{key.replace('_stats.json', '.tar')}",
                "nsamples": stats_data['successes']  # Using successes instead of count
            }
            shard_index['shardlist'].append(shard_entry)

    # Save the combined result to the root of the same bucket, formatted for readability
    s3.put_object(Bucket=bucket_name, Key=output_file, Body=json.dumps(shard_index, indent=4, sort_keys=True).encode('utf-8'))
    print(f"Generated shard index file saved to {bucket_name}/{output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <bucket_name> <prefix>")
        sys.exit(1)

    bucket_name = sys.argv[1]
    prefix = sys.argv[2]
    create_shard_index(bucket_name, prefix)

if __name__ == "__main__":
    main()
