import subprocess
import json
import sys

def run_command(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    if result.stderr:
        raise Exception("Error executing command: " + result.stderr)
    return result.stdout

def cancel_multipart_uploads(bucket_name):
    """Lists and cancels multipart uploads for the specified S3 bucket."""
    # List multipart uploads
    list_command = f"aws s3api list-multipart-uploads --bucket {bucket_name}"
    response = run_command(list_command)
    
    # Parse JSON response
    try:
        uploads = json.loads(response)
        if "Uploads" in uploads and uploads["Uploads"]:
            for upload in uploads["Uploads"]:
                upload_id = upload['UploadId']
                key = upload['Key']
                
                # Cancel each multipart upload
                cancel_command = f"aws s3api abort-multipart-upload --bucket {bucket_name} --key {key} --upload-id {upload_id}"
                print(f"Cancelling upload {upload_id} for key {key}")
                run_command(cancel_command)
                print(f"Cancelled upload {upload_id} for key {key}")
        else:
            print("No ongoing multipart uploads to cancel.")
    except json.JSONDecodeError:
        print("Failed to parse the JSON response from the AWS CLI.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <bucket_name>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    cancel_multipart_uploads(bucket_name)

if __name__ == "__main__":
    main()
