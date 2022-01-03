# google_image_scraper
About: A Web Scraping algorithm to retrieve top-n links of Google Image results using the Beautiful Soup and Selenium libraries. The algorithm takes in a user query, then preprocess it through the DataCleaner class to extract the keywords from the sentence. It also has a functionality to download the images URL to become a png image file to a designated directory.<br />

Note: The 10 images in the images directory results from calling `download_images_from_query("What is the first-difference filter?")`<br />

Functions:
* `download_images_from_query(query, num_image=10)`: **MAIN FUNCTION**. Call the other methods to download images from a user query. 
* `download_images_from_str_list(list_URL, dest_dir=".\images")`: iterate through the list of URL and use `urllib.request.urlretrieve` to download the images.
* `extract_filename_from_url(url, index)`: a helper function for `download_images_from_str_list()`. It takes in a URL, tries to extract the filename from the webpage, then strip any extension and replace it with .png.
* `get_image_list(query, num_image=10, max_load_time=3)`: A Web Scraping algorithm. Uses the DataCleaner's `preprocess_split_corpus()` to get the keywords in the query. Then it uses Selenium to search the processed query on Google Image, then add the URL of the images to a list to return.
* `preprocess_split_corpus()` (from data_cleaning.py - DataCleaner class): similar to the pre-processing text methods used in Natural Language Processing. The goal is to keep enough keywords to retain semantic of the original string.
* 1. Expand contractions (he's => he is)
* 2. Remove accented characters
* 3. Lowercase all characters
* 4. Lemmatize text
* 5. Remove special characters
* 6. Remove extra white spaces/lines
* 7. Remove stopwords
