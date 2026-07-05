import os
import sys
import shutil

# Add backend directory to python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

from app.core.database import SessionLocal, engine, Base
from app.services.chromadb import chromadb_service
from app.services.r2 import r2_service

# Import models
from app.models import User, Video, VideoMetadata, VideoScene, Job, Summary, QALog, SystemStat


def reset_db_keep_users():
    print("Clearing database records (preserving User accounts)...")
    db = SessionLocal()
    try:
        # Delete records from all tables EXCEPT User and VideoStandard
        db.query(QALog).delete()
        db.query(Summary).delete()
        db.query(VideoScene).delete()
        db.query(VideoMetadata).delete()
        db.query(Job).delete()
        db.query(SystemStat).delete()
        db.query(Video).delete()
        
        db.commit()
        print("Successfully cleared all data records from relational DB (preserved User accounts and Standards).")
    except Exception as e:
        db.rollback()
        print(f"Error resetting relational database: {e}")
    finally:
        db.close()


def clear_cloudflare_r2():
    print("Clearing Cloudflare R2 bucket objects...")
    # Force initial connection check
    r2_service.verify_connection()
    if r2_service.enabled:
        try:
            bucket_name = r2_service.bucket_name
            print(f"Listing objects in R2 bucket: {bucket_name}...")
            
            # Paginate through all objects
            paginator = r2_service.s3_client.get_paginator('list_objects_v2')
            deleted_count = 0
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                    print(f"Deleting {len(objects_to_delete)} objects from R2...")
                    r2_service.s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
                    deleted_count += len(objects_to_delete)
            print(f"Successfully cleared {deleted_count} objects in Cloudflare R2 bucket.")
        except Exception as r2_err:
            print(f"Error clearing R2 bucket: {r2_err}")
    else:
        print("Cloudflare R2 is disabled (mock mode). Skipping R2 cloud deletion.")


def reset_chromadb():
    print("Resetting ChromaDB vector database...")
    try:
        chromadb_service._ensure_connection()
        if chromadb_service.enabled and chromadb_service.client:
            try:
                chromadb_service.client.delete_collection(chromadb_service.collection_name)
                print(f"Deleted ChromaDB collection: {chromadb_service.collection_name}")
            except Exception as coll_err:
                print(f"Note: Collection did not exist or failed to delete: {coll_err}")
                
            chromadb_service.collection = chromadb_service.client.create_collection(chromadb_service.collection_name)
            print("Successfully recreated ChromaDB collection.")
        else:
            chromadb_service.mock_store = {}
            print("Cleared local mock ChromaDB store.")
    except Exception as e:
        print(f"Error resetting ChromaDB: {e}")


def clear_storage():
    print("Clearing local static mock storage files...")
    mock_r2_bucket = os.path.join(backend_dir, "storage", "mock_r2_bucket")
    if os.path.exists(mock_r2_bucket):
        try:
            for item in os.listdir(mock_r2_bucket):
                item_path = os.path.join(mock_r2_bucket, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print("Successfully cleared storage/mock_r2_bucket/ directory.")
        except Exception as e:
            print(f"Error clearing storage folder: {e}")
    else:
        print("Storage folder does not exist, skipping.")


if __name__ == "__main__":
    reset_db_keep_users()
    clear_cloudflare_r2()
    reset_chromadb()
    clear_storage()
    print("All databases and Cloudflare R2 file storage reset successfully (login accounts preserved)!")
