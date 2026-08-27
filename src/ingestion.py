import requests
import pandas as pd
import time
from pathlib import Path

def fetch_bird_metadata(species_name, max_pages=1):

    all_recordings=[]
    
    # API is paginated -> allows to loop through set number of pages  
    for page in range(1, max_pages+1):
        params = {
            'query': f'sp:"{species_name}" q:">D" len:"<120"',
            'key': "INSERT KEY",
            'page' : page
            }

        print(f"Fetching page {page} for {species_name}")
        response = requests.get('https://xeno-canto.org/api/3/recordings', params=params)

        if response.status_code == 200:
            data = response.json()
            # Retrieves recording object, defaults to empty list if not found
            recordings = data.get('recordings', [])
            all_recordings.extend(recordings)

            if page >= data.get('numPages', 1):
                break
        else:
            print(f"Status Code: {response.status_code}")
            print(f"URL Sent: {response.url}")
            print(f"Server Response: {response.text}")
            break

    if all_recordings: #  if it is non-empty
        df = pd.DataFrame(all_recordings)
        df = df[['id', 'en', 'gen', 'sp', 'type', 'q', 'length', 'file']]
        return df
    else:
        return pd.DataFrame()


def download_species_data(species_name, base_directory=None):
    df = fetch_bird_metadata(species_name)
    if df.empty:
        print(f"No recordings found for {species_name}")
        return
    
    # Creating File Path
    if base_directory is None:
        base_directory = Path.cwd()/ "data"
    else:
        base_directory = Path(base_directory)
    
    raw_directory = base_directory / 'raw'
    folder_name = species_name.replace(" ", "_").capitalize()
    species_directory = raw_directory / folder_name
    species_directory.mkdir(parents=True, exist_ok=True)

    # Adds a .csv save to the bird species folder with the species dataframe
    df.to_csv(species_directory / 'metadata.csv')

    # Returns tuple in form (id, en, file)
    for row in df[['id','folder_name','file', 'en']].itertuples(index=False):
        save_path = species_directory / f"{row.id}.mp3"

        # Checks if path already exists, ie. file has already been downloaded
        if save_path.exists():
            print(f'MP3 File #{row.id} already downloaded')
            continue

        # Downloads file using try/except for errors
        try:
            response = requests.get(row.file)

            if response.status_code == 200:
                save_path.write_bytes(response.content)
                print(f'MP3 File #{row.id} downloaded successfully')
                time.sleep(1)
            else:
                print(f'Failed to download MP3 File #{row.id}')
        except:
            print(f"Error downloading MP3 File #{row.id}: {row.en}")    