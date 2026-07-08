import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class R2Service:
    def __init__(self):
        self.bucket_name = settings.CF_R2_BUCKET_NAME
        self.public_url = settings.CF_R2_PUBLIC_URL
        self.mock_dir = os.path.join(os.getcwd(), "storage", "mock_r2_bucket")

        # Check if R2 configuration is complete, otherwise fallback to local mock mode
        self.enabled = all(
            [
                settings.CF_R2_ACCOUNT_ID,
                settings.CF_R2_ACCESS_KEY_ID,
                settings.CF_R2_SECRET_ACCESS_KEY,
            ]
        )

        if self.enabled:
            # R2 endpoint url format: https://<account_id>.r2.cloudflarestorage.com
            endpoint_url = f"https://{settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
            try:
                self.s3_client = boto3.client(
                    service_name="s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=settings.CF_R2_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.CF_R2_SECRET_ACCESS_KEY,
                    region_name="auto",  # R2 requires region_name='auto'
                )
                logger.info("Cloudflare R2 service initialized successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize boto3 R2 client: {e}. Falling back to Mock mode."
                )
                self.enabled = False
        else:
            logger.warning(
                "Cloudflare R2 credentials missing. Backend will run in local storage Mock mode."
            )
            if not os.path.exists(self.mock_dir):
                os.makedirs(self.mock_dir, exist_ok=True)

    def verify_connection(self) -> bool:
        """
        Verifies connection to Cloudflare R2 by performing a light check.
        If successful, returns True. Otherwise log warning/error and return False.
        """
        uvicorn_logger = logging.getLogger("uvicorn.error")
        if not self.enabled:
            uvicorn_logger.warning("Cloudflare R2 is disabled (Mock storage mode).")
            return False
        try:
            # Check bucket existence to verify credentials
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            uvicorn_logger.info("Connected to Cloudflare R2 successfully.")
            return True
        except Exception as e:
            uvicorn_logger.error(
                f"Cloudflare R2 connection verification failed: {e}. "
                "Falling back to local storage Mock mode."
            )
            self.enabled = False
            return False

    def upload_file(self, file_path: str, object_name: str) -> Optional[str]:
        """
        Uploads a file to Cloudflare R2 bucket.
        Returns the public URL if successful, otherwise None.
        """
        if self.enabled:
            try:
                self.s3_client.upload_file(
                    file_path, self.bucket_name, object_name
                )
                url = f"{self.public_url}/{object_name}"
                logger.info(f"File uploaded successfully to R2: {url}")
                return url
            except ClientError as e:
                logger.error(f"Failed to upload file to Cloudflare R2: {e}")
                return None
        else:
            # Mock mode: copy file to local mock directory
            import shutil

            try:
                dest_path = os.path.join(self.mock_dir, object_name)
                dest_dir = os.path.dirname(dest_path)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                # Return a local mock URL
                url = f"/static/mock_r2/{object_name}"
                logger.info(
                    f"[Mock R2] File stored locally at: {dest_path}. Mock URL: {url}"
                )
                return url
            except Exception as e:
                logger.error(f"[Mock R2] Failed to copy file: {e}")
                return None

    def get_presigned_url(self, object_name: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generates a presigned GET URL for a private R2 object.
        """
        if self.enabled:
            try:
                url = self.s3_client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket_name, "Key": object_name},
                    ExpiresIn=expires_in,
                )
                return url
            except Exception as e:
                logger.error(f"Failed to generate presigned URL for {object_name}: {e}")
                return None
        else:
            # In mock mode, return the local static path
            return f"/static/mock_r2/{object_name}"

    def delete_file(self, object_name: str) -> bool:
        """Deletes file from R2 or local mock storage."""
        if self.enabled:
            try:
                self.s3_client.delete_object(
                    Bucket=self.bucket_name, Key=object_name
                )
                logger.info(f"File {object_name} deleted from R2.")
                return True
            except ClientError as e:
                logger.error(f"Failed to delete file from R2: {e}")
                return False
        else:
            dest_path = os.path.join(self.mock_dir, object_name)
            if os.path.exists(dest_path):
                os.remove(dest_path)
                logger.info(f"[Mock R2] File {object_name} deleted from disk.")
                return True
            return False

    def extract_key(self, url_or_path: str) -> Optional[str]:
        """Extracts the R2/S3 object key from a URL or static path."""
        if not url_or_path:
            return None

        # 1. Handle mock URL prefix
        if url_or_path.startswith("/static/mock_r2/"):
            return url_or_path.replace("/static/mock_r2/", "", 1)

        # 2. Handle public URL prefix
        if self.public_url and url_or_path.startswith(self.public_url):
            key = url_or_path[len(self.public_url):]
            if key.startswith("/"):
                key = key[1:]
            return key

        # 3. Fallback: find standard folders in path
        for folder in ["videos/", "keyframes/"]:
            if folder in url_or_path:
                idx = url_or_path.find(folder)
                return url_or_path[idx:]

        return None


r2_service = R2Service()
