from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.ingestion import download_species_data
from src.processing import process_species_data

if __name__ == "__main__":
    # Ingestion Phase
    birds_to_download = ["cardinalis cardinalis", "cyanocitta cristata", "strix varia", "buteo jamaicensis", 'corvus brachyrhynchos']
    for bird in birds_to_download:
        download_species_data(bird)

    # Processing Phase
    project_data_directory = Path.cwd() / "data"
    mp3_files = (project_data_directory / "raw").rglob('*.mp3')

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_species_data, file, project_data_directory): file for file in mp3_files}

        for future in as_completed(future_to_file):
            original_file = future_to_file[future]
            try:
                result = future.result()
                print(f"Finished Processing: {original_file}")
            except Exception as exc:
                print(f"File {original_file} processing failed: {exc}")

    