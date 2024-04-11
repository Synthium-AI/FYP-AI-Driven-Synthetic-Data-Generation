#pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account_secret.json'
PARENT_FOLDER_ID = "1kLDYCBZowQWuzlTc94cfLRY1AJev3u9t"

class GoogleDriveAPI:
    def __init__(self):
        self.creds = self.authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def authenticate(self):
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return creds

    def upload_photo(self, file_path):
        try:
            file_metadata = {
                'name' : "alpha",
                'parents' : [PARENT_FOLDER_ID],
                "mimeType": "application/vnd.google-apps.folder"
            }

            file = self.service.files().create(
                body=file_metadata,
                # media_body=file_path
            ).execute()
            return True
        except Exception as e:
            print("[GoogleDriveAPI] Error uploading file: " + str(e))
            return False

if __name__ == '__main__':
    gda = GoogleDriveAPI()
    gda.upload_photo("test.png")