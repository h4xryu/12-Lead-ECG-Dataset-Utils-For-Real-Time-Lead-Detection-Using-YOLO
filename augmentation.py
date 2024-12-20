import os
import argparse
import cv2
from icrawler.builtin import GoogleImageCrawler

def crawl_image_data(key_word: str, max_num: int, des_dir: str):
    google_crawler = GoogleImageCrawler(storage={'root_dir': des_dir})
    google_crawler.crawl(keyword=key_word, max_num=max_num)

def remove_unreadable_images(des_dir: str):
    for root, _, files in os.walk(des_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    os.remove(file_path)
                    print(f"Removed unreadable image: {file_path}")
            except Exception as e:
                os.remove(file_path)
                print(f"Error reading image {file_path}, removed. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl images from Google based on a keyword.')
    parser.add_argument('--keyword', type=str, required=True, help='Keyword for image crawling')
    parser.add_argument('--max_num', type=int, default=100, help='Maximum number of images to download')
    parser.add_argument('--des_dir', type=str, default='./bg_noises', help='Destination directory for images')

    args = parser.parse_args()

    atempt_trial = 3
    while atempt_trial > 0:
        try:
            crawl_image_data(args.keyword, args.max_num, args.des_dir)
            remove_unreadable_images(args.des_dir)
        except Exception as e:
            print(f"An error occurred: {e}")
        atempt_trial -=1
