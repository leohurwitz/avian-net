from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
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

    # Dictionary to collect stats per species across processes
    species_stats = defaultdict(lambda: {'kept': 0, 'discarded': 0})

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_species_data, file, project_data_directory): file for file in mp3_files}

        for future in as_completed(future_to_file):
            original_file = future_to_file[future]
            try:
                result = future.result()
                species=result['species']
                species_stats[species]['kept'] += result['kept']
                species_stats[species]['discarded'] += result['discarded']
                print(f"Finished Processing: {original_file}")
            except Exception as exc:
                print(f"File {original_file} processing failed: {exc}")

    total_kept = 0
    total_discarded = 0

    for species, counts in species_stats.items():
        kept = counts['kept']
        discarded = counts['discarded']
        total = kept + discarded
        percent_discarded = (discarded / (total + 1e-7) * 100)

        total_kept += kept
        total_discarded += discarded
        print('=' * 50)
        print(f"Species: {species}")
        print(f"  - Kept Chunks:      {kept}")
        print(f"  - Discarded Chunks: {discarded} ({percent_discarded:.1f}% filtered out)")

    overall_total = total_kept + total_discarded
    overall_pct = (total_discarded / overall_total * 100) if overall_total > 0 else 0.0

    print("Overall Totals:")
    print(f"  - Total Kept Chunks:      {total_kept}")
    print(f"  - Total Discarded Chunks: {total_discarded} ({overall_pct:.1f}% filtered out)")
    print("=" * 50)

        

    