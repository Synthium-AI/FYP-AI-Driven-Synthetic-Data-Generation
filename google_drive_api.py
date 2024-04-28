#pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv, find_dotenv
import io
import os

load_dotenv(find_dotenv())

SCOPES = [os.getenv("GOOGLE_DRIVE_API_SCOPES")]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_DRIVE_API_SERVICE_ACCOUNT_FILE")
PARENT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_API_PARENT_FOLDER_ID")
CLIENT_BUFFER_FOLDER_NAME = os.getenv("CLIENT_BUFFER_FOLDER_NAME")

class GoogleDriveAPI:
    def __init__(self):
        self.creds = self.authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)
        if not os.path.exists(CLIENT_BUFFER_FOLDER_NAME):
            os.makedirs(CLIENT_BUFFER_FOLDER_NAME)

    def authenticate(self):
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return creds
        
    def upload_file(self, destination_folder_name, file_path):
        """Upload a file to the specified folder and prints file ID, folder ID
        Args: Name of the folder
        Returns: ID of the file uploaded"""
        try:
            # Get the folder ID for corresponding folder name
            folder_id = self.get_folder_id(destination_folder_name)
            if folder_id is None:
                print(f"[GoogleDriveAPI][ERROR] Folder '{destination_folder_name}' not found.")
                return False
            
            # Upload the file to Google Drive
            file_metadata = {"name": os.path.basename(file_path), "parents": [folder_id]}
            media = MediaFileUpload(
                file_path
            )
            # pylint: disable=maybe-no-member
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            print(f"[GoogleDriveAPI][SUCCESS] File '{file_path}' uploaded successfully with ID: {file.get('id')}")
            return file.get("id")

        except Exception as e:
            print("[GoogleDriveAPI] Error Uploading File: " + str(e))
            return False
        
    def download_file(self, parent_folder_name, file_name):
        """
        Downloads a file and saves it locally.

        Args:
            parent_folder_name: Name of the folder where the file is located.
            file_name: Name of the file to download.
            local_filename: Name of the local file to save the downloaded data.
        """
        try:
            local_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, file_name)
            file_id = self.get_file_id(parent_folder_name, file_name)
            if file_id is None:
                print(f"[GoogleDriveAPI][ERROR] File '{file_name}' not found in folder '{parent_folder_name}'.")
                return False

            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%")

            with open(local_file_path, 'wb') as local_file:
                local_file.write(file.getvalue())
            print(f"[GoogleDriveAPI][SUCCESS] File '{file_name}' downloaded successfully at {local_file_path}.")
            return local_file_path
        
        except Exception as e:
            print("[GoogleDriveAPI][ERROR] Error Downloading File: " + str(e))
            return False
        
    def delete_file(self, parent_folder_name, file_name):
        """Move specified file to the specified folder.
        Args:
            file_id: Id of the file to move.
            folder_id: Id of the folder
        Print: An object containing the new parent folder and other meta data
        Returns : Parent Ids for the file"""
        try:
            file_id = self.get_file_id(parent_folder_name, file_name)
            if file_id is None:
                print(f"[GoogleDriveAPI][ERROR] File '{file_name}' not found in folder '{parent_folder_name}'.")
                return False
            
            trash_folder_id = self.get_folder_id("trash")
            if trash_folder_id is None:
                print(f"[GoogleDriveAPI][ERROR] Trash Folder Not Found.")
                return False
            
            file = self.service.files().get(fileId=file_id, fields="parents").execute()
            previous_parents = ",".join(file.get("parents"))
            # Move the file to the new folder
            file = (
                self.service.files()
                .update(
                    fileId=file_id,
                    addParents=trash_folder_id,
                    removeParents=previous_parents,
                    fields="id, parents",
                )
                .execute()
            )
            print(f"[GoogleDriveAPI][SUCCESS] File '{file_name}' moved successfully to trash folder: '{trash_folder_id}'.")
            return True

        except Exception as e:
            print("[GoogleDriveAPI][ERROR] Error Deleting File: " + str(e))
            return False
        
    def get_folder_id(self, folder_name):
        try:
            files = []
            page_token = None
            while True:
                response = (
                    self.service.files()
                    .list(
                        q="'{}' in parents and mimeType='application/vnd.google-apps.folder'".format(PARENT_FOLDER_ID),
                        spaces="drive",
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break
            for file in files:
                if folder_name == file.get("name"):
                    print("[GoogleDriveAPI][SUCCESS] Folder '{}' found successfully with ID: {}".format(folder_name, file.get("id")))
                    return file.get("id")

        except Exception as e:
            print("[GoogleDriveAPI][ERROR] Error Getting Folder ID: " + str(e))

        print("[GoogleDriveAPI][ERROR] No Folder Corresponding to name '{}' was found!".format(folder_name))
        return None
    
    def get_file_id(self, parent_folder_name, file_name):
        try:
            parent_folder_id = self.get_folder_id(parent_folder_name)
            if parent_folder_id is None:
                print(f"[GoogleDriveAPI][ERROR] Folder '{parent_folder_name}' not found.")
                return None
            
            files = []
            page_token = None
            while True:
                response = (
                    self.service.files()
                    .list(
                        q="'{}' in parents and name = '{}'".format(parent_folder_id, file_name),
                        spaces="drive",
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break
            for file in files:
                if file_name == file.get("name"):
                    print("[GoogleDriveAPI][SUCCESS] File '{}' found successfully with ID: {}".format(file_name, file.get("id")))
                    return file.get("id")

        except Exception as e:
            print("[GoogleDriveAPI][ERROR] Error Getting File ID: " + str(e))

        print("[GoogleDriveAPI][ERROR] No File named '{}' was found in folder '{}' ".format(file_name, parent_folder_name))
        return None

if __name__ == '__main__':
    gda = GoogleDriveAPI()
    # gda.upload_photo("test.png")
    # folders = gda.get_folder_id("alpha")
    # print(folders)
    # gda.upload_file("model_configs","test.png")
    # gda.get_file_id("models", "good.jpg")
    # gda.download_file("models", "good.jpg")
    # gda.delete_file("models", "good.jpg")

