import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def download_tdf_2024_gpx():
    # URL containing the GPX links
    base_url = "https://www.cyclingstage.com/tour-de-france-2024-gpx/"

    # Fetch the page
    response = requests.get(base_url)
    response.raise_for_status()  # raise an error if the request failed

    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links that end with .gpx
    gpx_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".gpx"):
            # Convert a relative link to a full link if needed
            full_gpx_url = urljoin(base_url, href)
            gpx_links.append(full_gpx_url)

    # (Optional) Create a directory to save GPX files
    output_dir = "tdf_2024_gpx"
    os.makedirs(output_dir, exist_ok=True)

    # Download each GPX file
    for gpx_url in gpx_links:
        # Extract file name from URL
        file_name = os.path.basename(gpx_url)
        save_path = os.path.join(output_dir, file_name)

        print(f"Downloading {gpx_url} to {save_path}")
        file_resp = requests.get(gpx_url)
        file_resp.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(file_resp.content)

    print("All GPX files downloaded.")


if __name__ == "__main__":
    download_tdf_2024_gpx()
