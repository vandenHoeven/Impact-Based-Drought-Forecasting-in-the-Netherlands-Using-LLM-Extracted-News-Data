from playwright.sync_api import sync_playwright
import getpass
import time
import random
from datetime import datetime
import os
from pathlib import Path

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = str(CHAPTER_ROOT / "data" / "raw")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CURRENT_PAGE = 1


def sleep():
    sleep_time = random.uniform(4, 8)  # later change it to 5-15 seconds, 8 015
    sleep_time = 2

    print(f"Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)
    # print("Awake now, continuing...")


def check_boxes(page1):
    page1.get_by_role("checkbox", name="Hiermee selecteert u alle").check()


def next_page(page1):

    global CURRENT_PAGE
    next_btn = page1.locator('a[data-action="nextpage"]')
    next_btn.wait_for(state="visible", timeout=10000)
    next_btn.click()
    CURRENT_PAGE += 1
    print(f"Moved to page {CURRENT_PAGE}")
    sleep()


def download_boxes(page1, startpage, endpage):
    print(f"Downloading pages {startpage} to {endpage}...")
    page1.get_by_role("button", name="Downloaden").click()
    sleep()
    page1.get_by_role("textbox", name="Resultatenlijst voor_droogte").click()
    sleep()
    page1.get_by_role("textbox", name="Resultatenlijst voor_droogte").press("ControlOrMeta+a")
    sleep()
    page1.get_by_role("textbox", name="Resultatenlijst voor_droogte").fill(f"Page - {startpage} - {endpage}")
    sleep()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] expecting download...")
    with page1.expect_download(timeout=900000) as download_info:
        page1.get_by_role("contentinfo").get_by_role("button", name="Downloaden").click()
    print("succesfully initiated download...")
    download = download_info.value
    download_path = os.path.join(DOWNLOAD_DIR, f"Page_{startpage}_to_{endpage}.zip")
    download.save_as(download_path)
    print(f"Saved download to {download_path}")
    print("Download completed.")


def unselect_boxes(page1):
    page1.get_by_role("button", name="GESELECTEERD").click()
    sleep()
    page1.get_by_role("button", name="Alles wissen").click()
    sleep()
    page1.get_by_role("button", name="Verzendmap legen").click()
    sleep()
    sleep()
    sleep()
    page1.get_by_role("link", name="Vorige", exact=True).click()


def donwload_pages(page1, startpage, endpage):
    global CURRENT_PAGE
    while CURRENT_PAGE < startpage:
        next_page(page1)

    for p in range(startpage, endpage + 1):
        print(f"Processing page {p}...")
        check_boxes(page1)
        sleep()
        next_page(page1)
    download_boxes(page1, startpage, endpage)
    sleep()
    unselect_boxes(page1)
    sleep()


def run_lexis_scraper():
    username = input("Lexis username: ").strip()
    password = getpass.getpass("Lexis password: ")

    with sync_playwright() as p:
        # Launch browser (headless=False means you can see it working)
        browser = p.chromium.launch(headless=False)
        # context = browser.new_context() removed for accept downlaods
        context = browser.new_context(locale="nl-NL", accept_downloads=True)

        page = context.new_page()

        page.goto("https://www.lexisnexis.com/nl-nl/producten-login")
        page.get_by_role("button", name="Aanvullende cookies weigeren").click()
        # page.get_by_role("button", name="Reject Additional Cookies").click()

        sleep()
        with page.expect_popup() as page1_info:
            page.locator("#iw_comp1684473773185").get_by_role("link", name="Nexis Uni").click()
        page1 = page1_info.value
        sleep()
        # page1.get_by_role("textbox", name="ID").click() #swap sometimes idk why bug
        page1.get_by_role("textbox", name="Gebruikersnaam").click()
        print("clicked username textbox")
        sleep()
        page1.get_by_role("textbox", name="Gebruikersnaam").click()
        page1.get_by_role("textbox", name="Gebruikersnaam").dblclick()
        sleep()
        page1.get_by_role("textbox", name="Gebruikersnaam").fill(username)
        sleep()
        page1.get_by_role("button", name="Volgende").click()
        page1.get_by_role("textbox", name="Password").click()
        sleep()
        page1.get_by_role("textbox", name="Password").click()
        sleep()
        page1.get_by_role("textbox", name="Password").fill(password)
        sleep()
        page1.get_by_role("button", name="Sign In").click()
        sleep()
        # page1.goto("https://advance.lexis.com/bisnexishome/?pdmfid=1519360&crid=36f66902-8a14-45df-af4b-cb05f1714830")
        # sleep()
        page1.get_by_role("tab", name="Uitgebreid zoeken").click()
        sleep()
        page1.get_by_role("textbox", name="Geef een zoekterm op").click()
        sleep()
        page1.get_by_role("textbox", name="Geef een zoekterm op").fill(
            "droogte AND (impact OR landbouw OR opbrengst OR bosbrand OR scheepvaart OR drinkwater OR ziekte)"
        )
        sleep()
        page1.get_by_label("Zoeken", exact=True).click()
        sleep()
        page1.get_by_label("Sort-By").select_option("Datum: oudste eerst")
        sleep()
        page1.get_by_role("button", name=" Titelweergave").click()
        sleep()

        global CURRENT_PAGE
        # Run 1 bulk

        # Flexible bulk download loop
        START_PAGE = 0  # you can change this if you need to resume mid-way
        TOTAL_PAGES = 10  # total number of pages you want to download
        BLOCK_SIZE = 10  # always 10 pages per bulk exceeded daily downlaods at 10.34 --> Retry at 14.34
        start_time = time.time()
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Proces for pages {START_PAGE} to {TOTAL_PAGES}..."
        )

        for start in range(START_PAGE, TOTAL_PAGES + 1, BLOCK_SIZE):
            end = start + BLOCK_SIZE - 1
            if end > TOTAL_PAGES:
                end = TOTAL_PAGES
            print(f"\n=== Processing pages {start} to {end} ===\n")
            elapsed = time.time() - start_time
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step completed. Time elapsed: {elapsed:.2f} seconds"
            )
            donwload_pages(page1, startpage=start, endpage=end)
            sleep()
            sleep()
            sleep()
            sleep()
        print("All downloads completed.")
        elapsed = time.time() - start_time
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Download completed. Total time: {elapsed:.2f} seconds"
        )

        browser.close()


def run_lexis_viewer_smoke(duration_seconds: int = 20) -> None:
    """Open headed Chromium and exercise the Lexis login UI briefly (no real scrape)."""
    username = f"smoke_user_{random.randint(1000, 9999)}"
    password = f"smoke_pass_{random.randint(1000, 9999)}"
    print(f"Viewer smoke: dummy username={username!r}, duration={duration_seconds}s")

    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(locale="nl-NL", accept_downloads=True)
        page = context.new_page()

        try:
            page.goto("https://www.lexisnexis.com/nl-nl/producten-login")
            try:
                page.get_by_role("button", name="Aanvullende cookies weigeren").click(
                    timeout=5000
                )
            except Exception:
                pass

            sleep()
            with page.expect_popup() as page1_info:
                page.locator("#iw_comp1684473773185").get_by_role(
                    "link", name="Nexis Uni"
                ).click()
            page1 = page1_info.value
            sleep()

            page1.get_by_role("textbox", name="Gebruikersnaam").click()
            page1.get_by_role("textbox", name="Gebruikersnaam").fill(username)
            sleep()
            page1.get_by_role("button", name="Volgende").click()
            sleep()
            try:
                page1.get_by_role("textbox", name="Password").fill(password, timeout=5000)
            except Exception as exc:
                print(f"Viewer smoke: password field not filled ({exc})")
        except Exception as exc:
            print(f"Viewer smoke: UI step failed ({exc}); keeping browser open.")

        remaining = duration_seconds - (time.time() - started)
        if remaining > 0:
            print(f"Viewer smoke: holding browser open for {remaining:.1f}s more...")
            time.sleep(remaining)
        else:
            print("Viewer smoke: duration already elapsed; closing.")

        browser.close()
        print("Viewer smoke: done.")


if __name__ == "__main__":
    run_lexis_scraper()
