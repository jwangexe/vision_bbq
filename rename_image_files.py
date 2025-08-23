from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Optional progress bar

def rename_blob(bucket_name, blob, new_name, dry_run=True):
    """Rename a single blob with error handling"""
    try:
        if dry_run:
            return True, f"[DRY RUN] Would rename: {blob.name} -> {new_name}"
        
        # Create new client for thread safety
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob.name)  # Refresh blob reference
        
        # Copy blob to new location
        new_blob = bucket.copy_blob(blob, bucket, new_name)
        
        # Preserve metadata
        new_blob.content_type = blob.content_type
        new_blob.patch()
        
        # Delete original
        blob.delete()
        
        return True, f"Renamed: {blob.name} -> {new_name}"
    except Exception as e:
        return False, f"Error renaming {blob.name}: {str(e)}"

def add_jpg_extension_to_images(bucket_name, dry_run=True, max_workers=10):
    """Adds .jpg extension to image files in GCS bucket using concurrent processing"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # First pass: collect all blobs without extensions
    to_rename = []
    print("Scanning bucket for images without extensions...")
    for blob in tqdm(bucket.list_blobs()):
        filename = os.path.basename(blob.name)
        
        # Skip directories and files with extensions
        if not filename or '.' in filename:
            continue
            
        # Skip non-image files
        if not blob.content_type or not blob.content_type.startswith('image/'):
            continue
            
        to_rename.append(blob)
    
    print(f"Found {len(to_rename)} files to process")
    
    # Second pass: process collected blobs concurrently
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for blob in to_rename:
            new_name = f"{blob.name}.jpg"
            futures.append(
                executor.submit(
                    rename_blob, 
                    bucket_name, 
                    blob, 
                    new_name, 
                    dry_run
                )
            )
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            success, message = future.result()
            #print(message)
            if success:
                success_count += 1
            else:
                error_count += 1

    print(f"\nSummary:")
    print(f"Total files processed: {len(to_rename)}")
    print(f"Successfully renamed: {success_count}")
    print(f"Errors: {error_count}")
    if dry_run:
        print("DRY RUN COMPLETE - No changes made. Set dry_run=False to execute")

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    BUCKET_NAME = "visionbbq-images"  # Replace with your bucket name
    DRY_RUN = False  # Set to False to execute changes
    MAX_WORKERS = 20  # Adjust based on your system resources
    # =========================
    
    add_jpg_extension_to_images(
        bucket_name=BUCKET_NAME,
        dry_run=DRY_RUN,
        max_workers=MAX_WORKERS
    )